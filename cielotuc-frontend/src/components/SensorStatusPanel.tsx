import { useQuery } from "@tanstack/react-query";
import { API_BASE } from "../lib/api";
import type { SensorStatus } from "../lib/types";

export function SensorStatusPanel() {
  const { data: sensors, isLoading } = useQuery<SensorStatus[]>({
    queryKey: ["sensors-status"],
    queryFn: () =>
      fetch(`${API_BASE}/api/v1/sensors/status`).then((r) => r.json()),
    refetchInterval: 30000,
  });

  const online = sensors?.filter((s) => s.online).length ?? 0;
  const total = sensors?.length ?? 0;

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Sensores</h2>
        <span className="text-xs text-zinc-500">
          {online}/{total} en linea
        </span>
      </div>
      {isLoading ? (
        <p className="text-xs text-zinc-500">Cargando...</p>
      ) : !sensors?.length ? (
        <p className="text-xs text-zinc-500">Sin datos de sensores</p>
      ) : (
        <div className="space-y-2">
          {sensors.slice(0, 8).map((s) => (
            <div
              key={s.id}
              className="flex items-center justify-between rounded bg-zinc-800/50 px-3 py-1.5"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`h-2 w-2 rounded-full ${
                    s.online ? "bg-emerald-400" : "bg-zinc-600"
                  }`}
                />
                <span className="text-xs text-white">{s.name}</span>
              </div>
              <div className="flex items-center gap-3 text-[10px] text-zinc-500">
                {s.battery_pct != null && (
                  <span>{Math.round(s.battery_pct)}%</span>
                )}
                <span>{s.source}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
