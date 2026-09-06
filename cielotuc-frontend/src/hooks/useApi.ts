import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import { API_BASE } from "../lib/api";
import type {
  ModelMetrics,
  ComparisonResponse,
  SensorStatus,
  FloodAlertOut,
} from "../lib/types";

export function useModelMetrics() {
  return useQuery<ModelMetrics>({
    queryKey: ["model-metrics"],
    queryFn: async () => {
      const { data } = await axios.get<ModelMetrics>(
        `${API_BASE}/api/v1/model/metrics`
      );
      return data;
    },
    staleTime: 60 * 60 * 1000,
  });
}

export function useComparison(zoneId: number) {
  return useQuery<ComparisonResponse>({
    queryKey: ["comparison", zoneId],
    queryFn: async () => {
      const { data } = await axios.get<ComparisonResponse>(
        `${API_BASE}/api/v1/model/comparison`,
        { params: { zone_id: zoneId } }
      );
      return data;
    },
    staleTime: 60 * 60 * 1000,
  });
}

export function useSensorStatus() {
  return useQuery<SensorStatus[]>({
    queryKey: ["sensor-status"],
    queryFn: async () => {
      const { data } = await axios.get<SensorStatus[]>(
        `${API_BASE}/api/v1/sensors/status`
      );
      return data;
    },
    refetchInterval: 5 * 60 * 1000,
  });
}

export function useFloodHistory(limit = 50) {
  return useQuery<FloodAlertOut[]>({
    queryKey: ["flood-history", limit],
    queryFn: async () => {
      const { data } = await axios.get<FloodAlertOut[]>(
        `${API_BASE}/api/v1/alerts/flood/history`,
        { params: { limit } }
      );
      return data;
    },
    staleTime: 2 * 60 * 1000,
  });
}
