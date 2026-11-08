import { createRoot } from "react-dom/client";
import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";
import App from "./App";
import { Providers } from "./components/Providers";
createRoot(document.getElementById("root")!).render(<Providers><App /></Providers>);

import "./polish.css";
