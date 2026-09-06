import { Outlet, Navigate, Link, useLocation } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

const GOV_NAV = [
  { to: "/gobierno", label: "Dashboard" },
  { to: "/", label: "Sitio Ciudadano" },
];

export function GovLayout() {
  const { isAuthenticated, isGovernment, user, logout } = useAuth();
  const location = useLocation();

  if (!isAuthenticated || !isGovernment) {
    return <Navigate to="/login" replace />;
  }

  return (
    <div className="min-h-screen bg-zinc-950">
      <header className="sticky top-0 z-50 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-4">
            <Link to="/gobierno" className="flex items-center gap-2">
              <span className="text-lg font-bold text-white">CIELO·TUC</span>
              <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-medium text-amber-400">
                GOBIERNO
              </span>
            </Link>
            <nav className="flex gap-3">
              {GOV_NAV.map((l) => (
                <Link
                  key={l.to}
                  to={l.to}
                  className={`text-sm transition-colors ${
                    location.pathname === l.to
                      ? "text-white font-medium"
                      : "text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  {l.label}
                </Link>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs text-zinc-500">{user?.email}</span>
            <button
              onClick={logout}
              className="rounded bg-zinc-800 px-2 py-1 text-[10px] text-zinc-400 hover:bg-zinc-700"
            >
              Salir
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
