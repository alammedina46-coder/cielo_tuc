import type { CurrentConditions } from "../lib/types";

const CONDITION_ICONS: Record<string, string> = {
  clear: "☀️",
  partly_cloudy: "⛅",
  cloudy: "☁️",
  rain: "🌧️",
  heavy_rain: "⛈️",
  storm: "⛈️",
  thunderstorm: "⛈️",
  fog: "🌫️",
  wind: "💨",
  zonda: "🌬️",
  hail: "🧊",
  unknown: "🌡️",
};

interface Props {
  data: CurrentConditions;
  zoneName: string;
}

export function HeroWeather({ data, zoneName }: Props) {
  const icon = CONDITION_ICONS[data.condition] ?? "🌡️";
  const confidence = data.ai_confidence != null
    ? Math.round(data.ai_confidence * 100)
    : null;

  return (
    <div className="rounded-2xl bg-gradient-to-br from-sky-900/60 to-zinc-900 p-6 shadow-lg border border-sky-800/30">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-zinc-400 uppercase tracking-wide">
            {zoneName}
          </p>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-7xl font-bold tracking-tight text-white">
              {Math.round(data.temperature_c)}°
            </span>
            <span className="text-xl text-zinc-400">C</span>
          </div>
          <p className="mt-1 text-lg text-zinc-300 capitalize">
            {icon} {data.condition.replace(/_/g, " ")}
          </p>
        </div>

        {confidence !== null && (
          <div className="rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-medium text-emerald-400 border border-emerald-500/30">
            IA {confidence}% confianza
          </div>
        )}
      </div>

      <div className="mt-6 grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
        <Stat label="Sensación" value={`${Math.round(data.feels_like_c ?? data.temperature_c)}°`} />
        <Stat label="Humedad" value={`${Math.round(data.humidity_pct)}%`} />
        <Stat label="Viento" value={`${Math.round(data.wind_speed_kmh)} km/h`} />
        <Stat label="Presión" value={`${Math.round(data.pressure_hpa)} hPa`} />
      </div>

      {data.precip_24h_mm > 0 && (
        <div className="mt-4 rounded-lg bg-sky-800/30 px-4 py-2 text-sm text-sky-300">
          Precipitación últimas 24h: {data.precip_24h_mm.toFixed(1)} mm
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-white/5 px-3 py-2">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="font-medium text-zinc-200">{value}</p>
    </div>
  );
}
