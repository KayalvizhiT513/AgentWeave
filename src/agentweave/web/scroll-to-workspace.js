const workspace = document.querySelector("#workspace");

new MutationObserver(() => {
  if (!workspace.classList.contains("hidden")) {
    requestAnimationFrame(() => workspace.scrollIntoView({ behavior: "smooth", block: "start" }));
  }
}).observe(workspace, { attributes: true, attributeFilter: ["class"] });
