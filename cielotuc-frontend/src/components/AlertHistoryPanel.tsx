import { useQuery } from "@tanstack/react-query";
import { API_BASE } from "../lib/api";
import type { FloodAlertOut } from "../lib/types";

const SEVERITY_COLORS: Record<string, string> = {
  preventive: "bg-yellow-500/10 text-yellow-400",
  moderate: "bg-orange-500/10 text-orange-400",
  critical: "bg-red-500/10 text-red-400",
};

export function AlertHistoryPanel() {
  const { data: alerts, isLoading } = useQuery<FloodAlertOut[]>({
    queryKey: ["flood-history"],
    queryFn: () =>
      fetch(`${API_BASE}/api/v1/alerts/flood/history?limit=10`).then((r) =>
        r.json()
      ),
    refetchInterval: 60000,
  });

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
      <h2 className="mb-3 text-sm font-semibold text-white">
        Historial de Alertas
      </h2>
      {isLoading ? (
        <p className="text-xs text-zinc-500">Cargando...</p>
      ) : !alerts?.length ? (
        <p className="text-xs text-zinc-500">Sin alertas recientes</p>
      ) : (
        <div className="space-y-2">
          {alerts.map((a) => (
            <div
              key={a.id}
              className="flex items-center justify-between rounded bg-zinc-800/50 px-3 py-1.5"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                    SEVERITY_COLORS[a.severity] ?? "bg-zinc-700 text-zinc-400"
                  }`}
                >
                  {a.severity.toUpperCase()}
                </span>
                <span className="text-xs text-zinc-400">
                  Zona {a.zone_id}
                </span>
              </div>
              <div className="text-right">
                <div className="text-[10px] text-zinc-500">
                  {new Date(a.triggered_at).toLocaleDateString("es-AR")}
                </div>
                <div
                  className={`text-[10px] ${
                    a.delivered ? "text-emerald-400" : "text-zinc-500"
                  }`}
                >
                  {a.delivered ? "Enviada" : "Pendiente"}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
