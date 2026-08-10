const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const root = path.resolve(__dirname, "../../../..");
require(path.join(root, "reveal-game/js/domain.js"));
const Loader = require(path.join(root, "reveal-game/js/image-loader.js"));

test("nested relative path와 case-insensitive extension을 보존하고 자연 정렬한다", () => {
  const files = [
    { name: "image10.JPG", webkitRelativePath: "folder/sub/image10.JPG" },
    { name: "readme.txt", webkitRelativePath: "folder/readme.txt" },
    { name: "image2.png", webkitRelativePath: "folder/sub/image2.png" },
    { name: "image1.WeBp", webkitRelativePath: "folder/image1.WeBp" },
  ];
  const result = Loader.prepareFileEntries(files);
  assert.equal(result.ignoredCount, 1);
  assert.deepEqual(result.accepted.map((item) => item.relativePath), ["folder/image1.WeBp", "folder/sub/image2.png", "folder/sub/image10.JPG"]);
  assert.equal(new Set(result.accepted.map((item) => item.file)).size, 3);
});
