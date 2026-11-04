import { createRoot } from "react-dom/client";
import { startReactDsfr } from "@codegouvfr/react-dsfr/spa";
import { App } from "./App";
import "./docs.css";
import "./app.css";

startReactDsfr({ defaultColorScheme: "system" });

// No StrictMode: in development it runs each search twice, and "Approfondi" costs money.
createRoot(document.getElementById("root")!).render(<App />);
