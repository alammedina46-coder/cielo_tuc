import { BrowserRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "./lib/auth";
import { Layout } from "./components/Layout";
import { HomePage } from "./pages/HomePage";
import { ZoneDetailPage } from "./pages/ZoneDetailPage";
import { MapPage } from "./pages/MapPage";
import { ComparisonPage } from "./pages/ComparisonPage";
import { LoginPage } from "./pages/LoginPage";
import { GovDashboardPage } from "./pages/GovDashboardPage";
import { GovLayout } from "./components/GovLayout";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<HomePage />} />
              <Route path="zona/:id" element={<ZoneDetailPage />} />
              <Route path="mapa" element={<MapPage />} />
              <Route path="comparacion" element={<ComparisonPage />} />
            </Route>
            <Route path="login" element={<LoginPage />} />
            <Route element={<GovLayout />}>
              <Route path="gobierno" element={<GovDashboardPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
