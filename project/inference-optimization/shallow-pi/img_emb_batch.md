---
layout: post
title: "5. Valid Image Embedding Batching"
nav_exclude: true
section: project
subcategory: shallow-pi
date: 2026-06-22
tags:
  - Korean
  - Python
  - Profiling
language: ko
summary: "observation image를 각각 embed하지 말고 한번에 embedding을 구해서 나중에 split"
math: true
comments: true
comment_id: "project-shallow-pi-emb-batching"
permalink: /project/inference-optimization/shallow-pi/emb-batching/
---

<aside class="series-preface" markdown="1">

- **OS:** Ubuntu 22.04.5 LTS, Linux kernel 5.15.0
- **CPU:** 2 × AMD EPYC 9354 32-Core Processor
  - 64 physical cores / 128 threads total
- **Memory:** 503 GiB RAM
- **GPU:** 7 × NVIDIA L40S
  - ~48 GB VRAM per GPU
  - Used 1 out of 7 GPUs for inference & profiling
- **NVIDIA Driver / CUDA:** NVIDIA Driver 580.82.07 / CUDA 13.0
- **Storage:**
  - 2 TB NVMe SSD for the system drive
  - ~10.4 TB ext4 workspace/data storage
- **Usage context:** Shared multi-user research server
  - This machine is not a dedicated benchmark node.
  - Profiling was pinned to a single NVIDIA L40S, and that GPU was kept for this experiment during measurement.
  - CPU, system memory, storage I/O, and OS-level background load may still be affected by other users.

</aside>

이 서버는 여러 사용자가 함께 쓰는 공용 GPU 서버이므로, 전체 시스템을 독점한 dedicated benchmark 환경은 아니다. 따라서 이번 profiling에서는 실행 GPU를 1개의 L40S로 고정하고, 해당 GPU는 실험 중 단독으로 사용하기로 합의했다. 다만 CPU, memory, storage I/O, OS background load는 다른 사용자의 작업 영향을 받을 수 있으므로, 이후 latency 수치는 절대적인 서버 최대 성능이라기보다 shared-server 환경에서의 병목 분석용 측정값으로 해석한다.

---

[지난 게시물](/project/inference-optimization/shallow-pi/prefix-optim/)에서 사용하지 않는 image를 embedding하는 시간 낭비를 제거했었다. 하지만 남은 두개의 image는 여전히 각각 embed된다.

```text
image_0_embed ≈ 6.08 ms
image_1_embed ≈ 5.85 ms
```

즉 현재 방식은 `embed_image(base_0_rgb)`, `embed_image(left_wrist_0_rgb)`을 따로 구하는 것인데, 이번에 적용해볼 optimization은 `embed_image(cat([base_0_rgb, left_wrist_0_rgb], dim=0))`와 같이 embedding을 구하고 이것을 나중에 필요할 때 split하는 것이다. 목표는 아래와 같다:

```text
1. vision tower 호출 횟수 2회 → 1회
2. small-batch underutilization 완화
3. kernel launch / graph node overhead 감소
4. prefix token order는 동일하게 유지
```

핵심은 [`embed_prefix()`](https://github.com/icsl-Jeon/openpi/blob/a5940ac510c7f5d94918f238cfc3722be1a2c5c8/src/openpi/models_pytorch/pi0_pytorch.py#L186){:target="_blank" rel="noopener noreferrer"}의 image processing block을 아래 구조로 수정하는 것이다:

<details markdown="1">
<summary><code>pi0_pytorch.py</code>의 <code>embed_prefix()</code>에서 img processing block 수정</summary>
```shell
uv run python - <<'PY'
from pathlib import Path

path = Path("src/openpi/models_pytorch/pi0_pytorch.py")
text = path.read_text()

start = text.index("    def embed_prefix(")
end = text.index("    def embed_suffix(", start)

new_function = '''    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed image and language inputs for the PaliGemma prefix."""

        embs = []
        pad_masks = []
        att_masks = []

        # Collect image slots to be processed.
        image_items = []
        for image_idx, (img, img_mask) in enumerate(
            zip(images, img_masks, strict=True)
        ):
            # LIBERO pi0 uses a zero-filled, mask=False right-wrist slot.
            if (
                getattr(self, "libero_skip_right_wrist_image", False)
                and image_idx == 2
            ):
                continue

            image_items.append((image_idx, img, img_mask))

        def image_embed_func(img):
            return self.paligemma_with_expert.embed_image(img)

        use_batched_image_embed = (
            getattr(self, "libero_batch_valid_images", False)
            and not self.training
            and len(image_items) > 1
        )

        if use_batched_image_embed:
            batch_sizes = {int(img.shape[0]) for _, img, _ in image_items}
            if len(batch_sizes) != 1:
                raise ValueError(
                    f"All image slots must have the same batch size, got {batch_sizes}"
                )

            batch_size = next(iter(batch_sizes))

            # Preserve image-slot ordering:
            # [all samples of image_0, all samples of image_1, ...].
            batched_images = torch.cat(
                [img for _, img, _ in image_items],
                dim=0,
            )

            batched_image_embeddings = self._apply_checkpoint(
                image_embed_func,
                batched_images,
            )

            image_embeddings = torch.split(
                batched_image_embeddings,
                batch_size,
                dim=0,
            )

            for image_embedding, (_, _, image_mask) in zip(
                image_embeddings,
                image_items,
                strict=True,
            ):
                bsize, num_image_tokens = image_embedding.shape[:2]

                embs.append(image_embedding)
                pad_masks.append(
                    image_mask[:, None].expand(bsize, num_image_tokens)
                )
                att_masks += [0] * num_image_tokens

        else:
            for _, image, image_mask in image_items:
                image_embedding = self._apply_checkpoint(
                    image_embed_func,
                    image,
                )

                bsize, num_image_tokens = image_embedding.shape[:2]

                embs.append(image_embedding)
                pad_masks.append(
                    image_mask[:, None].expand(bsize, num_image_tokens)
                )
                att_masks += [0] * num_image_tokens

        # Language processing must remain unconditional and outside
        # both image-embedding branches.
        def language_embed_func(tokens):
            language_embedding = (
                self.paligemma_with_expert.embed_language_tokens(tokens)
            )
            embedding_dim = language_embedding.shape[-1]
            return language_embedding * math.sqrt(embedding_dim)

        language_embedding = self._apply_checkpoint(
            language_embed_func,
            lang_tokens,
        )

        embs.append(language_embedding)
        pad_masks.append(lang_masks)

        num_language_tokens = language_embedding.shape[1]
        att_masks += [0] * num_language_tokens

        prefix_embeddings = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)

        prefix_attention_masks = torch.tensor(
            att_masks,
            dtype=torch.bool,
            device=prefix_pad_masks.device,
        )

        bsize = prefix_pad_masks.shape[0]
        prefix_attention_masks = prefix_attention_masks[None, :].expand(
            bsize,
            len(att_masks),
        )

        return (
            prefix_embeddings,
            prefix_pad_masks,
            prefix_attention_masks,
        )

'''

path.write_text(text[:start] + new_function + text[end:])
print("Replaced embed_prefix() in:", path)
PY
```
{: style="margin-left: 1rem;" }

</details>

실행 결과는 아래와 같다:

| 상태                        |  Model-only median |      Policy median |
| ------------------------- | -----------------: | -----------------: |
| `while→for` + image2 skip |          15.886 ms |          18.487 ms |
| + valid-image batching    |      **14.425 ms** |      **16.992 ms** |
| 추가 감소                     | **1.461 ms, 9.2%** | **1.496 ms, 8.1%** |


최초 baseline과 비교하면 아래과 같다:

```text
Model-only: 22.270 ms → 14.425 ms
            35.2% latency reduction, 1.54× speedup

Policy:     23.741 ms → 16.992 ms
            28.4% latency reduction, 1.40× speedup
```


{% comment %}{% include comments.html %}{% endcomment %}
