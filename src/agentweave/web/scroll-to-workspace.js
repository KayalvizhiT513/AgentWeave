const workspace = document.querySelector("#workspace");
const dashboardGrid = document.querySelector(".dashboard-grid");

new MutationObserver(() => {
  if (!workspace.classList.contains("hidden")) {
    requestAnimationFrame(() => dashboardGrid.scrollIntoView({ behavior: "smooth", block: "start" }));
  }
}).observe(workspace, { attributes: true, attributeFilter: ["class"] });
