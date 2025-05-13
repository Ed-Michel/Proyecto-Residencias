window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clients: {
        bloquearTeclado: function(value) {
            let ids = ["damping", "preference", "threshold", "n_clusters", "eps", "min_samples"];
            ids.forEach(function(id) {
                let el = document.getElementById(id);
                if (el && !el.hasListenerAttached) {
                    el.addEventListener("keydown", function(event) {
                        if (!["ArrowUp", "ArrowDown", "Tab", "Shift", "Control"].includes(event.key)) {
                            event.preventDefault();
                        }
                    });
                    el.hasListenerAttached = true;
                }
            });
            return window.dash_clientside.no_update;
        }
    }
});