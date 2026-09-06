import { useState } from "react";
import { useAuth } from "../hooks/useAuth";
import { SensorStatusPanel } from "../components/SensorStatusPanel";
import { AlertHistoryPanel } from "../components/AlertHistoryPanel";
import { ModelMetricsPanel } from "../components/ModelMetricsPanel";
import { SendAlertForm } from "../components/SendAlertForm";

type Tab = "overview" | "sensors" | "alerts" | "model";

const TABS: { key: Tab; label: string }[] = [
  { key: "overview", label: "Resumen" },
  { key: "sensors", label: "Sensores" },
  { key: "alerts", label: "Alertas" },
  { key: "model", label: "Modelo IA" },
];

export function GovDashboardPage() {
  const { user } = useAuth();
  const [tab, setTab] = useState<Tab>("overview");

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Panel de Gobierno</h1>
          <p className="text-xs text-zinc-500">
            {user?.name} — {user?.role === "admin" ? "Administrador" : "Técnico"}
          </p>
        </div>
        <span className="rounded bg-emerald-500/10 px-2 py-1 text-[10px] font-medium text-emerald-400">
          EN LINEA
        </span>
      </div>

      <div className="flex gap-1 border-b border-zinc-800">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-sm transition-colors ${
              tab === t.key
                ? "border-b-2 border-sky-500 text-white"
                : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "overview" && <OverviewTab />}
      {tab === "sensors" && <SensorStatusPanel />}
      {tab === "alerts" && (
        <div className="grid gap-6 lg:grid-cols-2">
          <SendAlertForm />
          <AlertHistoryPanel />
        </div>
      )}
      {tab === "model" && <ModelMetricsPanel />}
    </div>
  );
}

function OverviewTab() {
  return (
    <div className="space-y-6">
      <ModelMetricsPanel />
      <div className="grid gap-6 lg:grid-cols-2">
        <SensorStatusPanel />
        <AlertHistoryPanel />
      </div>
    </div>
  );
}
