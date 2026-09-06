import { Outlet, Link, useLocation } from "react-router-dom";

const NAV_LINKS = [
  { to: "/", label: "Inicio" },
  { to: "/mapa", label: "Mapa" },
  { to: "/comparacion", label: "Comparación" },
];

export function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-50 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-4xl items-center justify-between px-4 py-3">
          <Link to="/" className="flex items-center gap-2">
            <span className="text-lg font-bold text-white">CIELO·TUC</span>
            <span className="rounded bg-sky-500/20 px-1.5 py-0.5 text-[10px] font-medium text-sky-400">
              IA
            </span>
          </Link>
          <nav className="flex gap-4 items-center">
            {NAV_LINKS.map((l) => (
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
            <Link
              to="/login"
              className="rounded bg-zinc-800 px-2 py-1 text-[10px] text-zinc-400 hover:bg-zinc-700 hover:text-zinc-300"
            >
              Gobierno
            </Link>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-4xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
