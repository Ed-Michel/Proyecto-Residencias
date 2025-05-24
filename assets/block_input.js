window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clients: {
        validarTeclado: function(value) {
            let ids = ["damping", "preference", "threshold", "n_clusters", "eps", "min_samples"];
            const validKeys = [
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "-", ".", "Backspace", "Delete", "ArrowLeft", "ArrowRight", 
                "ArrowUp", "ArrowDown", "Tab", "Enter", "Home", "End"
            ];

            ids.forEach(function(id) {
                let el = document.getElementById(id);
                if (el && !el.hasListenerAttached) {
                    el.addEventListener("keydown", function(event) {
                        if (!validKeys.includes(event.key)) {
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