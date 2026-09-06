import type { DailyForecast } from "../lib/types";

interface Props {
  daily: DailyForecast[];
}

export function ForecastDaily({ daily }: Props) {
  if (!daily.length) return null;

  return (
    <div className="rounded-xl bg-zinc-900/60 p-4 border border-zinc-800">
      <h3 className="mb-3 text-sm font-semibold text-zinc-400 uppercase tracking-wide">
        Pronóstico 7 días
      </h3>
      <div className="space-y-2">
        {daily.map((d) => {
          const prob = Math.round(d.rain_probability * 100);
          const tempRange = Math.round(d.temp_max_c - d.temp_min_c);
          return (
            <div
              key={d.date}
              className="flex items-center gap-3 rounded-lg bg-white/5 px-3 py-2"
            >
              <span className="w-24 text-sm font-medium text-zinc-300">
                {d.day_name}
              </span>
              <div className="flex-1 flex items-center gap-2">
                <span className="text-xs text-zinc-500 w-8 text-right">
                  {Math.round(d.temp_min_c)}°
                </span>
                <div className="flex-1 h-2 rounded-full bg-zinc-700 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-sky-500 to-amber-500"
                    style={{ width: `${Math.min(tempRange * 8, 100)}%` }}
                  />
                </div>
                <span className="text-xs text-zinc-500 w-8">
                  {Math.round(d.temp_max_c)}°
                </span>
              </div>
              <span
                className={`w-12 text-center text-xs font-medium rounded-full px-2 py-0.5 ${
                  prob < 20
                    ? "text-emerald-400 bg-emerald-500/10"
                    : prob < 50
                      ? "text-amber-400 bg-amber-500/10"
                      : "text-red-400 bg-red-500/10"
                }`}
              >
                {prob}%
              </span>
              {d.zonda_risk > 0.5 && (
                <span className="text-xs text-amber-400" title="Riesgo Zonda">🌬️</span>
              )}
              {d.storm_risk > 0.5 && (
                <span className="text-xs text-red-400" title="Riesgo tormenta">⛈️</span>
              )}
              {d.hail_risk > 0.5 && (
                <span className="text-xs text-blue-300" title="Riesgo granizo">🧊</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
