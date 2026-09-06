export interface ZoneOut {
  id: number;
  name: string;
  department: string;
  latitude: number;
  longitude: number;
  altitude_m: number | null;
  population: number | null;
  is_mountain: boolean;
  impermeable_pct: number | null;
}

export interface CurrentConditions {
  zone_id: number;
  zone_name: string;
  timestamp: string;
  temperature_c: number;
  feels_like_c: number | null;
  humidity_pct: number;
  pressure_hpa: number;
  wind_speed_kmh: number;
  wind_direction_deg: number | null;
  wind_gust_kmh: number | null;
  precip_1h_mm: number;
  precip_24h_mm: number;
  visibility_km: number | null;
  uv_index: number | null;
  cloud_cover_pct: number | null;
  condition: string;
  ai_confidence: number | null;
  zonda_risk_score: number | null;
}

export interface HourlyForecast {
  target_time: string;
  horizon_hours: number;
  temperature_c: number;
  rain_probability: number;
  precip_mm_expected: number;
  wind_speed_kmh: number | null;
  condition: string;
  confidence: number;
}

export interface DailyForecast {
  date: string;
  day_name: string;
  condition: string;
  temp_min_c: number;
  temp_max_c: number;
  rain_probability: number;
  precip_mm_expected: number;
  ai_confidence: number;
  hail_risk: number;
  zonda_risk: number;
  storm_risk: number;
}

export interface ForecastResponse {
  zone_id: number;
  zone_name: string;
  generated_at: string;
  model_version: string;
  current: CurrentConditions;
  hourly: HourlyForecast[];
  daily: DailyForecast[];
}

export interface ZondaIndex {
  timestamp: string;
  risk_score: number;
  risk_level: "none" | "low" | "medium" | "high" | "active";
  cordillera_pressure_hpa: number | null;
  thermal_differential_c: number | null;
  descending_speed_kmh: number | null;
  forecast_24h_probability: number;
}

export interface ModelMetrics {
  version: string;
  trained_at: string;
  training_samples: number | null;
  accuracy_overall: number | null;
  precision_rain: number | null;
  recall_rain: number | null;
  rmse_temperature: number | null;
  avg_lead_time_hours: number | null;
  false_positive_rate: number | null;
  is_active: boolean;
  accuracy_history: Array<{ version: string; accuracy: number }>;
}

export interface ComparisonRow {
  variable: string;
  cielotuc: string;
  smn: string;
  weather_com: string;
}

export interface ComparisonResponse {
  zone_id: number;
  generated_at: string;
  rows: ComparisonRow[];
  cielotuc_accuracy: number;
  smn_accuracy: number;
  weathercom_accuracy: number;
  notable_wins: Array<Record<string, unknown>>;
}

export interface SensorStatus {
  id: number;
  code: string;
  name: string;
  source: string;
  is_active: boolean;
  battery_pct: number | null;
  last_seen: string | null;
  online: boolean;
}

export interface FloodAlertOut {
  id: number;
  zone_id: number;
  severity: string;
  rain_probability: number;
  expected_precip_mm: number | null;
  trigger_type: string;
  notes: string | null;
  triggered_at: string;
  delivered: boolean;
  floodtuc_response_code: number | null;
  outcome: string | null;
}

export interface UserOut {
  id: number;
  email: string;
  name: string;
  role: "ciudadano" | "tecnico" | "admin";
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: UserOut;
}

export interface NotificationSend {
  to: string;
  message: string;
  channels: string[];
}

export interface NotificationResult {
  channel: string;
  to: string;
  sent: boolean;
  message_sid: string | null;
  error: string | null;
}

export interface NotificationSendResponse {
  status: string;
  results: NotificationResult[];
}
