import { useState, type FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { API_BASE } from "../lib/api";
import { useAuth } from "../hooks/useAuth";
import type { NotificationSendResponse } from "../lib/types";

export function SendAlertForm() {
  const { token } = useAuth();
  const [phone, setPhone] = useState("");
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<NotificationSendResponse | null>(null);

  const send = useMutation({
    mutationFn: async () => {
      const res = await fetch(`${API_BASE}/api/v1/notifications/send`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          to: phone,
          message,
          channels: ["sms"],
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Error al enviar");
      }
      return res.json() as Promise<NotificationSendResponse>;
    },
    onSuccess: (data) => {
      setResult(data);
      setMessage("");
    },
  });

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    setResult(null);
    send.mutate();
  };

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
      <h2 className="mb-3 text-sm font-semibold text-white">
        Enviar Alerta SMS
      </h2>
      <form onSubmit={handleSubmit} className="space-y-3">
        <input
          type="tel"
          placeholder="+5493815551234"
          value={phone}
          onChange={(e) => setPhone(e.target.value)}
          required
          className="w-full rounded border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-white placeholder-zinc-500 focus:border-sky-500 focus:outline-none"
        />
        <textarea
          placeholder="Mensaje de alerta meteorologica..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          required
          rows={3}
          className="w-full rounded border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-white placeholder-zinc-500 focus:border-sky-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={send.isPending || !phone || !message}
          className="w-full rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50"
        >
          {send.isPending ? "Enviando..." : "Enviar SMS"}
        </button>
      </form>

      {result && (
        <div className="mt-3 space-y-1">
          {result.results.map((r, i) => (
            <div
              key={i}
              className={`rounded px-3 py-1.5 text-xs ${
                r.sent
                  ? "bg-emerald-500/10 text-emerald-400"
                  : "bg-red-500/10 text-red-400"
              }`}
            >
              {r.sent
                ? `Enviado a ${r.to} via ${r.channel}`
                : `Error ${r.channel}: ${r.error}`}
            </div>
          ))}
        </div>
      )}

      {send.isError && (
        <div className="mt-2 rounded bg-red-500/10 px-3 py-1.5 text-xs text-red-400">
          {send.error?.message || "Error al enviar notificacion"}
        </div>
      )}
    </div>
  );
}
