import type { ZondaIndex } from "../lib/types";

interface Props {
  data: ZondaIndex;
}

const RISK_COLORS: Record<string, string> = {
  none: "#10b981",
  low: "#34d399",
  medium: "#fbbf24",
  high: "#f59e0b",
  active: "#ef4444",
};

const RISK_LABELS: Record<string, string> = {
  none: "Sin riesgo",
  low: "Bajo",
  medium: "Medio",
  high: "Alto",
  active: "Activo",
};

export function ZondaGauge({ data }: Props) {
  const score = Math.round(data.risk_score);
  const angle = (score / 100) * 180;
  const color = RISK_COLORS[data.risk_level] ?? "#71717a";
  const label = RISK_LABELS[data.risk_level] ?? data.risk_level;
  const forecast24 = Math.round(data.forecast_24h_probability * 100);

  return (
    <div className="rounded-xl bg-zinc-900/60 p-5 border border-zinc-800">
      <h3 className="mb-2 text-sm font-semibold text-zinc-400 uppercase tracking-wide">
        Monitor Zonda
      </h3>
      <div className="flex flex-col items-center">
        <svg viewBox="0 0 200 110" className="w-48 h-auto">
          <defs>
            <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="40%" stopColor="#fbbf24" />
              <stop offset="70%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="url(#gaugeGrad)"
            strokeWidth="12"
            strokeLinecap="round"
          />
          <line
            x1="100"
            y1="100"
            x2={100 + 60 * Math.cos(((180 - angle) * Math.PI) / 180)}
            y2={100 - 60 * Math.sin(((180 - angle) * Math.PI) / 180)}
            stroke={color}
            strokeWidth="3"
            strokeLinecap="round"
          />
          <circle cx="100" cy="100" r="5" fill={color} />
        </svg>
        <div className="mt-2 text-center">
          <span className="text-3xl font-bold" style={{ color }}>
            {score}
          </span>
          <span className="text-sm text-zinc-500"> / 100</span>
        </div>
        <span
          className="mt-1 rounded-full px-3 py-0.5 text-xs font-medium"
          style={{ backgroundColor: `${color}20`, color }}
        >
          {label}
        </span>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
        <div className="rounded-lg bg-white/5 px-3 py-2">
          <p className="text-zinc-500">Presión Cordillera</p>
          <p className="font-medium text-zinc-300">
            {data.cordillera_pressure_hpa != null
              ? `${Math.round(data.cordillera_pressure_hpa)} hPa`
              : "—"}
          </p>
        </div>
        <div className="rounded-lg bg-white/5 px-3 py-2">
          <p className="text-zinc-500">Δ Térmico</p>
          <p className="font-medium text-zinc-300">
            {data.thermal_differential_c != null
              ? `${data.thermal_differential_c.toFixed(1)}°C`
              : "—"}
          </p>
        </div>
        <div className="rounded-lg bg-white/5 px-3 py-2">
          <p className="text-zinc-500">Vel. descenso</p>
          <p className="font-medium text-zinc-300">
            {data.descending_speed_kmh != null
              ? `${Math.round(data.descending_speed_kmh)} km/h`
              : "—"}
          </p>
        </div>
        <div className="rounded-lg bg-white/5 px-3 py-2">
          <p className="text-zinc-500">Prob. 24h</p>
          <p className="font-medium text-zinc-300">{forecast24}%</p>
        </div>
      </div>
    </div>
  );
}
