import { useState } from "react";
import { useZones } from "../hooks/useZones";
import { useComparison } from "../hooks/useApi";
import { ComparisonTable } from "../components/ComparisonTable";

export function ComparisonPage() {
  const { data: zones } = useZones();
  const [zoneId, setZoneId] = useState(1);
  const { data: comparison, isLoading } = useComparison(zoneId);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-white">Comparación de precisión</h1>
        {zones && (
          <select
            value={zoneId}
            onChange={(e) => setZoneId(Number(e.target.value))}
            className="rounded-lg bg-zinc-800 px-3 py-1.5 text-sm text-zinc-300 border border-zinc-700"
          >
            {zones.map((z) => (
              <option key={z.id} value={z.id}>
                {z.name}
              </option>
            ))}
          </select>
        )}
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-20">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-600 border-t-sky-400" />
        </div>
      ) : comparison ? (
        <ComparisonTable data={comparison} />
      ) : (
        <div className="rounded-xl bg-zinc-900/60 p-8 text-center text-zinc-500 border border-zinc-800">
          Sin datos de comparación disponibles
        </div>
      )}
    </div>
  );
}
