import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { API_BASE } from "../lib/api";
import type { ForecastResponse, ZondaIndex } from "../lib/types";

export function useForecast(zoneId: number | null) {
  return useQuery<ForecastResponse>({
    queryKey: ["forecast", zoneId],
    queryFn: async () => {
      const { data } = await axios.get<ForecastResponse>(
        `${API_BASE}/api/v1/forecast/${zoneId}`
      );
      return data;
    },
    enabled: zoneId !== null,
    refetchInterval: 30 * 60 * 1000,
  });
}

export function useZondaIndex(zoneId: number | null) {
  return useQuery<ZondaIndex>({
    queryKey: ["zonda", zoneId],
    queryFn: async () => {
      const { data } = await axios.get<ZondaIndex>(
        `${API_BASE}/api/v1/forecast/${zoneId}/zonda`
      );
      return data;
    },
    enabled: zoneId !== null,
    refetchInterval: 30 * 60 * 1000,
  });
}
