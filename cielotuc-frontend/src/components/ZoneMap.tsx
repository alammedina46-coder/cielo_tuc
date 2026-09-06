import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import { useNavigate } from "react-router-dom";
import type { ZoneOut } from "../lib/types";
import "leaflet/dist/leaflet.css";

interface Props {
  zones: ZoneOut[];
}

export function ZoneMap({ zones }: Props) {
  const navigate = useNavigate();

  return (
    <div className="rounded-xl overflow-hidden border border-zinc-800">
      <MapContainer
        center={[-26.82, -65.22]}
        zoom={10}
        className="h-[400px] w-full"
        zoomControl={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />
        {zones.map((z) => (
          <Marker
            key={z.id}
            position={[z.latitude, z.longitude]}
            eventHandlers={{
              click: () => navigate(`/zona/${z.id}`),
            }}
          >
            <Popup>
              <div className="text-sm">
                <p className="font-semibold">{z.name}</p>
                <p className="text-gray-500">{z.department}</p>
                {z.altitude_m != null && (
                  <p className="text-xs text-gray-400">{z.altitude_m} msnm</p>
                )}
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
}
