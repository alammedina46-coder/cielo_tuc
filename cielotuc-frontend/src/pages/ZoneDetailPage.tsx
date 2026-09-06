import { useParams, Link } from "react-router-dom";
import { useZone } from "../hooks/useZones";
import { useForecast, useZondaIndex } from "../hooks/useForecast";
import { HeroWeather } from "../components/HeroWeather";
import { ForecastHourly } from "../components/ForecastHourly";
import { ForecastDaily } from "../components/ForecastDaily";
import { ZondaGauge } from "../components/ZondaGauge";

export function ZoneDetailPage() {
  const { id } = useParams<{ id: string }>();
  const zoneId = id ? Number(id) : null;

  const { data: zone, isLoading: loadingZone } = useZone(zoneId);
  const { data: forecast, isLoading: loadingForecast } = useForecast(zoneId);
  const { data: zonda } = useZondaIndex(zoneId);

  if (loadingZone || loadingForecast) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="flex items-center gap-3 text-zinc-400">
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-600 border-t-sky-400" />
          <span className="text-sm">Cargando datos de la zona...</span>
        </div>
      </div>
    );
  }

  if (!zone || !forecast) {
    return (
      <div className="py-20 text-center text-zinc-500">
        Zona no encontrada.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-300 transition-colors"
      >
        ← Volver
      </Link>

      <div className="rounded-xl bg-zinc-900/40 px-4 py-3 border border-zinc-800 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">{zone.name}</h1>
          <p className="text-sm text-zinc-400">{zone.department}</p>
        </div>
        {zone.altitude_m != null && (
          <span className="text-xs text-zinc-500 bg-zinc-800 rounded-full px-3 py-1">
            {zone.altitude_m} msnm
          </span>
        )}
      </div>

      <HeroWeather data={forecast.current} zoneName={zone.name} />
      <ForecastHourly hourly={forecast.hourly} />
      <ForecastDaily daily={forecast.daily} />
      {zonda && <ZondaGauge data={zonda} />}
    </div>
  );
}
