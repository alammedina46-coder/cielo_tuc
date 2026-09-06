import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { API_BASE } from "../lib/api";
import { useAuth } from "../hooks/useAuth";
import type { ModelMetrics } from "../lib/types";

export function ModelMetricsPanel() {
  const { token } = useAuth();
  const queryClient = useQueryClient();

  const { data: metrics, isLoading } = useQuery<ModelMetrics>({
    queryKey: ["model-metrics"],
    queryFn: () =>
      fetch(`${API_BASE}/api/v1/model/metrics`).then((r) => r.json()),
  });

  const retrain = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${API_BASE}/api/v1/model/retrain`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!res.ok) throw new Error("Error al reentrenar");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["model-metrics"] });
    },
  });

  if (isLoading) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
        <p className="text-xs text-zinc-500">Cargando metricas del modelo...</p>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
        <p className="text-xs text-zinc-500">No hay modelo activo</p>
      </div>
    );
  }

  const accPct = metrics.accuracy_overall != null
    ? `${(metrics.accuracy_overall * 100).toFixed(1)}%`
    : "N/A";
  const rainPct = metrics.precision_rain != null
    ? `${(metrics.precision_rain * 100).toFixed(1)}%`
    : "N/A";
  const rmse = metrics.rmse_temperature != null
    ? `${metrics.rmse_temperature.toFixed(1)}°C`
    : "N/A";

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">
          Modelo IA — v{metrics.version}
        </h2>
        <div className="flex items-center gap-2">
          <span className="rounded bg-sky-500/10 px-1.5 py-0.5 text-[10px] text-sky-400">
            {metrics.is_active ? "ACTIVO" : "INACTIVO"}
          </span>
          <button
            onClick={() => retrain.mutate()}
            disabled={retrain.isPending}
            className="rounded bg-zinc-800 px-2 py-1 text-[10px] text-zinc-400 hover:bg-zinc-700 disabled:opacity-50"
          >
            {retrain.isPending ? "Reentrenando..." : "Reentrenar"}
          </button>
        </div>
      </div>
      <div className="mb-3 text-[10px] text-zinc-500">
        Entrenado: {new Date(metrics.trained_at).toLocaleDateString("es-AR")}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <KpiCard label="Precision Global" value={accPct} />
        <KpiCard label="Precision Lluvia" value={rainPct} />
        <KpiCard label="RMSE Temperatura" value={rmse} />
      </div>

      {metrics.accuracy_history.length > 1 && (
        <div className="mt-3">
          <p className="mb-1 text-[10px] text-zinc-500">Historial de versiones</p>
          <div className="flex gap-1">
            {metrics.accuracy_history.map((h) => (
              <div
                key={h.version}
                className="rounded bg-zinc-800 px-2 py-1 text-center"
              >
                <div className="text-[9px] text-zinc-500">v{h.version}</div>
                <div className="text-[10px] text-white">
                  {(h.accuracy * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {retrain.isSuccess && (
        <div className="mt-2 rounded bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-400">
          Reentrenamiento encolado (task: {retrain.data?.task_id})
        </div>
      )}
      {retrain.isError && (
        <div className="mt-2 rounded bg-red-500/10 px-3 py-1.5 text-xs text-red-400">
          Error al reentrenar modelo
        </div>
      )}
    </div>
  );
}

function KpiCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-zinc-800/50 p-3 text-center">
      <div className="text-lg font-bold text-white">{value}</div>
      <div className="text-[10px] text-zinc-500">{label}</div>
    </div>
  );
}
