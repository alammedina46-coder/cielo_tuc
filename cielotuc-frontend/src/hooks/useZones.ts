import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { API_BASE } from "../lib/api";
import type { ZoneOut } from "../lib/types";

export function useZones() {
  return useQuery<ZoneOut[]>({
    queryKey: ["zones"],
    queryFn: async () => {
      const { data } = await axios.get<ZoneOut[]>(
        `${API_BASE}/api/v1/zones/`
      );
      return data;
    },
    staleTime: 10 * 60 * 1000,
  });
}

export function useZone(id: number | null) {
  return useQuery<ZoneOut>({
    queryKey: ["zone", id],
    queryFn: async () => {
      const { data } = await axios.get<ZoneOut>(
        `${API_BASE}/api/v1/zones/${id}`
      );
      return data;
    },
    enabled: id !== null,
    staleTime: 10 * 60 * 1000,
  });
}
