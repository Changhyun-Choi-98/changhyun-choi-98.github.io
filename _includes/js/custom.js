(function () {
  var storageKey = "sidebar-expanded-items";

  function onReady(callback) {
    if (window.jtd && window.jtd.onReady) {
      window.jtd.onReady(callback);
    } else if (document.readyState !== "loading") {
      callback();
    } else {
      document.addEventListener("DOMContentLoaded", callback);
    }
  }

  function nav() {
    return document.getElementById("site-nav");
  }

  function normalizePath(path) {
    var parser = document.createElement("a");
    parser.href = path;

    var pathname = parser.pathname || "/";
    if (pathname.length > 1 && pathname.charAt(pathname.length - 1) !== "/") {
      pathname += "/";
    }

    return pathname;
  }

  function itemKey(item) {
    var link = item.querySelector(":scope > .nav-list-link");
    if (link) {
      var href = link.getAttribute("href");
      if (href) {
        return normalizePath(href);
      }

      if (link.classList.contains("active")) {
        return normalizePath(window.location.pathname);
      }

      if (link.href) {
        return normalizePath(link.href);
      }
    }

    var category = item.querySelector(":scope > .nav-category");
    if (category) {
      return "category:" + category.textContent.trim();
    }

    return null;
  }

  function expandedKeys() {
    var siteNav = nav();
    if (!siteNav) {
      return [];
    }

    return Array.prototype.map
      .call(siteNav.querySelectorAll(".nav-list-item.active"), itemKey)
      .filter(Boolean);
  }

  function saveExpandedState() {
    try {
      window.sessionStorage.setItem(storageKey, JSON.stringify(expandedKeys()));
    } catch (error) {}
  }

  function setExpanded(item) {
    if (!item) {
      return;
    }

    item.classList.add("active");

    var expander = item.querySelector(":scope > .nav-list-expander");
    if (expander) {
      expander.setAttribute("aria-expanded", "true");
    }
  }

  function expandForLink(link, markLink) {
    if (!link) {
      return false;
    }

    if (markLink) {
      link.classList.add("active");
    }

    var target = link;
    while (target) {
      while (target && !(target.classList && target.classList.contains("nav-list-item"))) {
        target = target.parentNode;
      }

      if (target) {
        setExpanded(target);
        target = target.parentNode;
      }
    }

    return true;
  }

  function findLinkByPath(pathname) {
    var siteNav = nav();
    if (!siteNav) {
      return null;
    }

    var normalizedPath = normalizePath(pathname);
    var links = siteNav.querySelectorAll(".nav-list-link[href]");

    for (var index = 0; index < links.length; index += 1) {
      var link = links[index];
      if (normalizePath(link.getAttribute("href")) === normalizedPath) {
        return link;
      }
    }

    return null;
  }

  function expandClosestAncestor() {
    var pathname = normalizePath(window.location.pathname);
    var segments = pathname.split("/").filter(Boolean);

    while (segments.length > 0) {
      segments.pop();

      var ancestorPath = "/" + segments.join("/") + "/";
      var ancestorLink = findLinkByPath(ancestorPath);
      if (ancestorLink) {
        return expandForLink(ancestorLink, true);
      }
    }

    return false;
  }

  function restoreExpandedState() {
    var siteNav = nav();
    if (!siteNav) {
      return false;
    }

    var keys;
    try {
      keys = JSON.parse(window.sessionStorage.getItem(storageKey) || "[]");
    } catch (error) {
      keys = [];
    }

    if (!Array.isArray(keys) || keys.length === 0) {
      return false;
    }

    var restored = false;
    siteNav.querySelectorAll(".nav-list-item").forEach(function (item) {
      if (keys.indexOf(itemKey(item)) !== -1) {
        setExpanded(item);
        restored = true;
      }
    });

    return restored;
  }

  onReady(function () {
    var siteNav = nav();
    if (!siteNav) {
      return;
    }

    var restored = restoreExpandedState();
    var hasActiveLink = Boolean(siteNav.querySelector(".nav-list-link.active"));
    var openedAncestor = hasActiveLink ? false : expandClosestAncestor();

    if (restored || openedAncestor || expandedKeys().length > 0) {
      saveExpandedState();
    }

    siteNav.addEventListener("click", function (event) {
      var target = event.target;
      while (target && !(target.classList && target.classList.contains("nav-list-expander"))) {
        target = target.parentNode;
      }

      if (target) {
        window.setTimeout(saveExpandedState, 0);
      }
    });
  });
})();
