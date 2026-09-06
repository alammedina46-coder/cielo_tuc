import type { HourlyForecast } from "../lib/types";

interface Props {
  hourly: HourlyForecast[];
}

function probabilityColor(prob: number): string {
  if (prob < 0.2) return "bg-emerald-500";
  if (prob < 0.4) return "bg-emerald-400";
  if (prob < 0.6) return "bg-amber-400";
  if (prob < 0.8) return "bg-amber-500";
  return "bg-red-500";
}

export function ForecastHourly({ hourly }: Props) {
  if (!hourly.length) {
    return (
      <div className="rounded-xl bg-zinc-900/60 p-4 border border-zinc-800">
        <h3 className="mb-3 text-sm font-semibold text-zinc-400 uppercase tracking-wide">
          Pronóstico horario — próximas 12h
        </h3>
        <p className="text-zinc-500 text-sm">Sin datos disponibles</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-zinc-900/60 p-4 border border-zinc-800">
      <h3 className="mb-3 text-sm font-semibold text-zinc-400 uppercase tracking-wide">
        Pronóstico horario — próximas 12h
      </h3>
      <div className="flex gap-2 overflow-x-auto pb-2">
        {hourly.map((h) => {
          const hour = new Date(h.target_time).getHours();
          const prob = Math.round(h.rain_probability * 100);
          const barHeight = Math.max(4, prob * 0.8);
          return (
            <div
              key={h.target_time}
              className="flex flex-col items-center gap-1 min-w-[56px]"
            >
              <span className="text-xs text-zinc-400">
                {String(hour).padStart(2, "0")}h
              </span>
              <span className="text-sm font-medium text-zinc-200">
                {Math.round(h.temperature_c)}°
              </span>
              <div
                className={`w-6 rounded-full ${probabilityColor(h.rain_probability)}`}
                style={{ height: `${barHeight}px`, minHeight: "4px" }}
              />
              <span className="text-[10px] text-zinc-500">{prob}%</span>
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex items-center gap-3 text-[10px] text-zinc-500">
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-500" /> Baja
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-amber-400" /> Media
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-red-500" /> Alta
        </span>
      </div>
    </div>
  );
}
