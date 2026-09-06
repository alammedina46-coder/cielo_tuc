import { useState } from "react";
import { useZones } from "../hooks/useZones";
import { useForecast, useZondaIndex } from "../hooks/useForecast";
import { HeroWeather } from "../components/HeroWeather";
import { ForecastHourly } from "../components/ForecastHourly";
import { ForecastDaily } from "../components/ForecastDaily";
import { ZondaGauge } from "../components/ZondaGauge";

export function HomePage() {
  const { data: zones, isLoading: loadingZones } = useZones();
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const { data: forecast, isLoading: loadingForecast } = useForecast(selectedId);
  const { data: zonda } = useZondaIndex(selectedId);

  const defaultZone = zones?.[0];
  const activeId = selectedId ?? defaultZone?.id ?? null;
  const activeZone = zones?.find((z) => z.id === activeId);

  if (loadingZones) {
    return <LoadingState text="Cargando zonas..." />;
  }

  if (!zones?.length) {
    return (
      <div className="text-center py-20 text-zinc-500">
        No se encontraron zonas disponibles.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 overflow-x-auto pb-2">
        {zones.slice(0, 10).map((z) => (
          <button
            key={z.id}
            onClick={() => setSelectedId(z.id)}
            className={`whitespace-nowrap rounded-full px-4 py-1.5 text-sm transition-all ${
              activeId === z.id
                ? "bg-sky-600 text-white shadow-md"
                : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
            }`}
          >
            {z.name}
          </button>
        ))}
      </div>

      {loadingForecast && activeId ? (
        <LoadingState text="Cargando pronóstico..." />
      ) : forecast ? (
        <>
          <HeroWeather data={forecast.current} zoneName={activeZone?.name ?? forecast.zone_name} />
          <ForecastHourly hourly={forecast.hourly} />
          <ForecastDaily daily={forecast.daily} />
        </>
      ) : (
        <div className="rounded-xl bg-zinc-900/60 p-8 text-center text-zinc-500 border border-zinc-800">
          Seleccioná una zona para ver el pronóstico
        </div>
      )}

      {zonda && <ZondaGauge data={zonda} />}
    </div>
  );
}

function LoadingState({ text }: { text: string }) {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="flex items-center gap-3 text-zinc-400">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-600 border-t-sky-400" />
        <span className="text-sm">{text}</span>
      </div>
    </div>
  );
}
