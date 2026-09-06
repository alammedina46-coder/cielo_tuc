import { useZones } from "../hooks/useZones";
import { ZoneMap } from "../components/ZoneMap";

export function MapPage() {
  const { data: zones, isLoading } = useZones();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-zinc-600 border-t-sky-400" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold text-white">Mapa de microclimas</h1>
      <p className="text-sm text-zinc-400">
        17 departamentos de Tucumán — hacé click en una zona para ver su pronóstico.
      </p>
      {zones && <ZoneMap zones={zones} />}
    </div>
  );
}
