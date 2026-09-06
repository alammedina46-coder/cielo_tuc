import type { ComparisonResponse } from "../lib/types";

interface Props {
  data: ComparisonResponse;
}

export function ComparisonTable({ data }: Props) {
  return (
    <div className="rounded-xl bg-zinc-900/60 p-4 border border-zinc-800">
      <h3 className="mb-3 text-sm font-semibold text-zinc-400 uppercase tracking-wide">
        Comparación con pronósticos oficiales
      </h3>

      <div className="grid grid-cols-3 gap-3 mb-4 text-center">
        <AccuracyBadge
          label="CIELO·TUC"
          value={data.cielotuc_accuracy}
          color="emerald"
        />
        <AccuracyBadge
          label="SMN"
          value={data.smn_accuracy}
          color="amber"
        />
        <AccuracyBadge
          label="Weather.com"
          value={data.weathercom_accuracy}
          color="zinc"
        />
      </div>

      {data.rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-700">
                <th className="py-2 text-left text-zinc-500">Variable</th>
                <th className="py-2 text-emerald-400">CIELO·TUC</th>
                <th className="py-2 text-amber-400">SMN</th>
                <th className="py-2 text-zinc-400">Weather.com</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((r) => (
                <tr key={r.variable} className="border-b border-zinc-800/50">
                  <td className="py-2 text-zinc-300">{r.variable}</td>
                  <td className="py-2 text-center text-zinc-300">{r.cielotuc}</td>
                  <td className="py-2 text-center text-zinc-400">{r.smn}</td>
                  <td className="py-2 text-center text-zinc-400">{r.weather_com}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function AccuracyBadge({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: "emerald" | "amber" | "zinc";
}) {
  const colors = {
    emerald: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
    amber: "bg-amber-500/10 text-amber-400 border-amber-500/30",
    zinc: "bg-zinc-500/10 text-zinc-400 border-zinc-500/30",
  };
  return (
    <div className={`rounded-lg border px-3 py-2 ${colors[color]}`}>
      <p className="text-[10px] uppercase tracking-wider">{label}</p>
      <p className="text-xl font-bold">{Math.round(value * 100)}%</p>
    </div>
  );
}
