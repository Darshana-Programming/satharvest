
"use client";

import { useEffect, useMemo, useState } from "react";

type YieldRecord = {
  County: string;
  Year: number;
  Value: number | null;
  "CV (%)": number | null;
};

type WeatherRecord = {
  County: string;
  Year: number;
  avg_temp: number | null;
  total_precipitation: number | null;
  hot_days_30: number | null;
  longest_dry_spell: number | null;
  avg_humidity: number | null;
  avg_solar: number | null;
  early_avg_temp: number | null;
  early_total_precipitation: number | null;
  mid_avg_temp: number | null;
  mid_total_precipitation: number | null;
  late_avg_temp: number | null;
  late_total_precipitation: number | null;
};

type MetricKey =
  | "avg_temp"
  | "total_precipitation"
  | "hot_days_30"
  | "longest_dry_spell"
  | "avg_humidity";

type Season = "early" | "mid" | "late";

type Pair = {
  year: number;
  yieldValue: number;
  weatherValue: number;
};

const metrics: {
  key: MetricKey;
  label: string;
  unit: string;
}[] = [
  { key: "avg_temp", label: "Average Temperature", unit: "°C" },
  { key: "total_precipitation", label: "Total Precipitation", unit: "mm" },
  { key: "hot_days_30", label: "Hot Days Above 30°C", unit: "days" },
  { key: "longest_dry_spell", label: "Longest Dry Spell", unit: "days" },
  { key: "avg_humidity", label: "Average Humidity", unit: "%" },
];

const seasons: { key: Season; label: string }[] = [
  { key: "early", label: "Early (April–May)" },
  { key: "mid", label: "Mid (June–July)" },
  { key: "late", label: "Late (August–September)" },
];

function normalizeCounty(value: string) {
  return value.trim().toUpperCase();
}

function validNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function StatCard({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800 p-5">
      <p className="text-sm text-slate-400">{label}</p>
      <p className="mt-2 text-2xl font-bold text-green-400">
        {value}
      </p>
    </div>
  );
}

function BarChart({
  data,
  unit,
  color = "green",
}: {
  data: { year: number; value: number | null }[];
  unit: string;
  color?: "green" | "blue" | "amber";
}) {
  const values = data
    .map((r) => r.value)
    .filter(validNumber);

  if (!values.length) {
    return (
      <p className="mt-5 text-slate-400">
        No available observations.
      </p>
    );
  }

  const max = Math.max(...values, 0);
  const min = Math.min(...values, 0);
  const range = max - min || 1;

  const colors = {
    green: "bg-green-500",
    blue: "bg-sky-500",
    amber: "bg-amber-500",
  };

  return (
    <div className="mt-6 overflow-x-auto">
      <div className="flex h-72 min-w-[580px] items-end gap-3 border-b border-slate-600 pb-2">
        {data.map((r) => {
          const height = validNumber(r.value)
            ? ((r.value - min) / range) * 85 + 5
            : 0;

          return (
            <div
              key={r.year}
              className="flex h-full min-w-12 flex-1 flex-col items-center justify-end gap-2"
            >
              <span className="text-xs text-slate-300">
                {validNumber(r.value)
                  ? r.value.toFixed(1)
                  : "N/A"}
              </span>
              <div
                className={`w-full max-w-16 rounded-t ${colors[color]}`}
                style={{ height: `${height}%` }}
                title={`${r.year}: ${r.value ?? "N/A"} ${unit}`}
              />
              <span className="text-xs text-slate-400">
                {r.year}
              </span>
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-xs text-slate-500">
        Bar heights are scaled to the displayed data range.
        Exact values appear above each bar.
      </p>
    </div>
  );
}

function pearsonCorrelation(pairs: Pair[]): number | null {
  if (pairs.length < 2) return null;

  const n = pairs.length;
  const meanX =
    pairs.reduce((s, p) => s + p.weatherValue, 0) / n;
  const meanY =
    pairs.reduce((s, p) => s + p.yieldValue, 0) / n;

  let numerator = 0;
  let sumX2 = 0;
  let sumY2 = 0;

  for (const p of pairs) {
    const dx = p.weatherValue - meanX;
    const dy = p.yieldValue - meanY;

    numerator += dx * dy;
    sumX2 += dx * dx;
    sumY2 += dy * dy;
  }

  if (sumX2 === 0 || sumY2 === 0) return null;

  return numerator / Math.sqrt(sumX2 * sumY2);
}

function ScatterPlot({
  pairs,
  weatherLabel,
  unit,
}: {
  pairs: Pair[];
  weatherLabel: string;
  unit: string;
}) {
  if (!pairs.length) {
    return (
      <p className="mt-6 text-slate-400">
        No paired observations available.
      </p>
    );
  }

  const xValues = pairs.map((p) => p.weatherValue);
  const yValues = pairs.map((p) => p.yieldValue);

  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);

  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const left = 65;
  const right = 640;
  const top = 25;
  const bottom = 285;

  const scaleX = (x: number) =>
    left + ((x - xMin) / xRange) * (right - left);

  const scaleY = (y: number) =>
    bottom - ((y - yMin) / yRange) * (bottom - top);

  return (
    <div className="mt-6 overflow-x-auto">
      <svg
        viewBox="0 0 700 365"
        className="min-w-[550px] w-full"
        role="img"
        aria-label={`Scatter plot of ${weatherLabel} against corn yield`}
      >
        <line
          x1={left}
          y1={top}
          x2={left}
          y2={bottom}
          stroke="#94a3b8"
        />
        <line
          x1={left}
          y1={bottom}
          x2={right}
          y2={bottom}
          stroke="#94a3b8"
        />

        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const y = bottom - fraction * (bottom - top);
          const value = yMin + fraction * yRange;

          return (
            <g key={fraction}>
              <line
                x1={left}
                y1={y}
                x2={right}
                y2={y}
                stroke="#475569"
                strokeDasharray="3 5"
              />
              <text
                x={left - 8}
                y={y + 4}
                fill="#cbd5e1"
                fontSize="11"
                textAnchor="end"
              >
                {value.toFixed(1)}
              </text>
            </g>
          );
        })}

        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const x = left + fraction * (right - left);
          const value = xMin + fraction * xRange;

          return (
            <text
              key={fraction}
              x={x}
              y={bottom + 20}
              fill="#cbd5e1"
              fontSize="11"
              textAnchor="middle"
            >
              {value.toFixed(1)}
            </text>
          );
        })}

        {pairs.map((p) => (
          <circle
            key={p.year}
            cx={scaleX(p.weatherValue)}
            cy={scaleY(p.yieldValue)}
            r="5"
            fill="#22c55e"
            stroke="#bbf7d0"
            strokeWidth="1"
          >
            <title>
              {`${p.year}: ${p.weatherValue.toFixed(2)} ${unit}, ${p.yieldValue.toFixed(2)} bu/acre`}
            </title>
          </circle>
        ))}

        <text
          x="350"
          y="345"
          fill="#e2e8f0"
          fontSize="13"
          textAnchor="middle"
        >
          {`${weatherLabel} (${unit})`}
        </text>

        <text
          transform="translate(16 155) rotate(-90)"
          fill="#e2e8f0"
          fontSize="13"
          textAnchor="middle"
        >
          Corn Yield (bu/acre)
        </text>
      </svg>
      <p className="text-xs text-slate-400">
        Each point represents one matched county-year observation.
        Hover over a point to see its values.
      </p>
    </div>
  );
}

export default function Home() {
  const [yields, setYields] = useState<YieldRecord[]>([]);
  const [weather, setWeather] = useState<WeatherRecord[]>([]);
  const [county, setCounty] = useState("BOONE");
  const [metric, setMetric] =
    useState<MetricKey>("total_precipitation");
  const [season, setSeason] = useState<Season>("mid");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadData() {
      try {
        const [yieldResponse, weatherResponse] = await Promise.all([
          fetch("/data/corn-yield.json"),
          fetch("/data/weather-features.json"),
        ]);

        if (!yieldResponse.ok || !weatherResponse.ok) {
          throw new Error("Unable to load required JSON datasets.");
        }

        const yieldData = await yieldResponse.json();
        const weatherData = await weatherResponse.json();

        if (!Array.isArray(yieldData) || !Array.isArray(weatherData)) {
          throw new Error("Unexpected JSON data format.");
        }

        setYields(yieldData);
        setWeather(weatherData);
      } catch (e) {
        setError(
          e instanceof Error ? e.message : "Data loading failed."
        );
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  const counties = useMemo(
    () =>
      [...new Set(yields.map((r) => normalizeCounty(r.County)))]
        .sort(),
    [yields]
  );

  const countyYields = useMemo(
    () =>
      yields
        .filter(
          (r) =>
            normalizeCounty(r.County) === county &&
            validNumber(r.Value)
        )
        .sort((a, b) => a.Year - b.Year),
    [yields, county]
  );

  const countyWeather = useMemo(
    () =>
      weather
        .filter((r) => normalizeCounty(r.County) === county)
        .sort((a, b) => a.Year - b.Year),
    [weather, county]
  );

  const weatherLookup = useMemo(
    () => new Map(countyWeather.map((r) => [r.Year, r])),
    [countyWeather]
  );

  const pairs = useMemo(() => {
    const result: Pair[] = [];

    for (const r of countyYields) {
      const w = weatherLookup.get(r.Year);

      if (
        w &&
        validNumber(r.Value) &&
        validNumber(w[metric])
      ) {
        result.push({
          year: r.Year,
          yieldValue: r.Value,
          weatherValue: w[metric],
        });
      }
    }

    return result;
  }, [countyYields, weatherLookup, metric]);

  const selectedMetric = metrics.find((m) => m.key === metric)!;
  const correlation = pearsonCorrelation(pairs);

  const latest = countyYields.at(-1);

  const average =
    countyYields.length > 0
      ? countyYields.reduce(
          (sum, r) => sum + Number(r.Value),
          0
        ) / countyYields.length
      : null;

  const seasonalTempKey =
    `${season}_avg_temp` as keyof WeatherRecord;

  const seasonalRainKey =
    `${season}_total_precipitation` as keyof WeatherRecord;

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <header className="border-b border-slate-800 px-6 py-7">
        <div className="mx-auto max-w-6xl">
          <h1 className="text-4xl font-bold text-green-400">
            SatHarvest
          </h1>
          <p className="mt-2 text-slate-300">
            From Space to Soil | NASA Weather and Missouri Corn Yields
          </p>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-6 py-10">
        <h2 className="text-3xl font-bold">
          Missouri Corn Yield and Weather Explorer
        </h2>

        <p className="mt-4 max-w-3xl text-slate-300">
          Explore historical Missouri corn yields alongside
          NASA POWER weather data. Compare weather indicators
          with observed agricultural outcomes.
        </p>

        {loading && (
          <p className="mt-6">Loading datasets...</p>
        )}

        {error && (
          <p className="mt-6 text-red-400">{error}</p>
        )}

        {!loading && !error && (
          <>
            <div className="mt-8 grid gap-4 md:grid-cols-3">
              <StatCard
                label="Corn Yield Records"
                value={String(yields.length)}
              />
              <StatCard
                label="Counties with Yield Records"
                value={String(counties.length)}
              />
              <StatCard
                label="NASA Weather Records"
                value={String(weather.length)}
              />
            </div>

            <section className="mt-10 rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h3 className="text-2xl font-semibold text-green-400">
                County Explorer
              </h3>

              <label
                htmlFor="county"
                className="mt-5 block text-sm text-slate-300"
              >
                Select Missouri County
              </label>

              <select
                id="county"
                value={county}
                onChange={(e) => setCounty(e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-600 bg-slate-800 p-3 md:w-80"
              >
                {counties.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>

              <div className="mt-7 grid gap-4 md:grid-cols-2">
                <StatCard
                  label="Latest Available Yield"
                  value={
                    latest && validNumber(latest.Value)
                      ? `${latest.Value.toFixed(1)} bu/acre (${latest.Year})`
                      : "N/A"
                  }
                />
                <StatCard
                  label="Average Recorded Yield"
                  value={
                    average === null
                      ? "N/A"
                      : `${average.toFixed(1)} bu/acre`
                  }
                />
              </div>

              <h4 className="mt-9 text-xl font-semibold">
                Historical Corn Yield — {county}
              </h4>

              <BarChart
                data={countyYields.map((r) => ({
                  year: r.Year,
                  value: r.Value,
                }))}
                unit="bu/acre"
              />
            </section>

            <section className="mt-10 rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h3 className="text-2xl font-semibold text-green-400">
                NASA POWER Weather Analysis
              </h3>

              <label
                htmlFor="metric"
                className="mt-5 block text-sm text-slate-300"
              >
                Select Weather Indicator
              </label>

              <select
                id="metric"
                value={metric}
                onChange={(e) =>
                  setMetric(e.target.value as MetricKey)
                }
                className="mt-2 w-full rounded-lg border border-slate-600 bg-slate-800 p-3 md:w-80"
              >
                {metrics.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                  </option>
                ))}
              </select>

              <h4 className="mt-8 text-xl font-semibold">
                {selectedMetric.label} — {county}
              </h4>

              <BarChart
                data={countyWeather.map((r) => ({
                  year: r.Year,
                  value: r[metric],
                }))}
                unit={selectedMetric.unit}
                color="blue"
              />
            </section>

            <section className="mt-10 rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h3 className="text-2xl font-semibold text-green-400">
                Growing-Season Weather Analysis
              </h3>

              <label
                htmlFor="season"
                className="mt-5 block text-sm text-slate-300"
              >
                Growing-Season Period
              </label>

              <select
                id="season"
                value={season}
                onChange={(e) =>
                  setSeason(e.target.value as Season)
                }
                className="mt-2 w-full rounded-lg border border-slate-600 bg-slate-800 p-3 md:w-80"
              >
                {seasons.map((s) => (
                  <option key={s.key} value={s.key}>
                    {s.label}
                  </option>
                ))}
              </select>

              <h4 className="mt-8 text-xl font-semibold">
                Seasonal Average Temperature
              </h4>

              <BarChart
                data={countyWeather.map((r) => ({
                  year: r.Year,
                  value: r[seasonalTempKey] as number | null,
                }))}
                unit="°C"
                color="amber"
              />

              <h4 className="mt-9 text-xl font-semibold">
                Seasonal Total Precipitation
              </h4>

              <BarChart
                data={countyWeather.map((r) => ({
                  year: r.Year,
                  value: r[seasonalRainKey] as number | null,
                }))}
                unit="mm"
                color="blue"
              />
            </section>

            <section className="mt-10 rounded-2xl border border-green-800 bg-slate-900 p-6">
              <h3 className="text-2xl font-semibold text-green-400">
                Weather vs. Corn Yield Analysis
              </h3>

              <p className="mt-3 text-slate-300">
                This comparison uses matching county-year records
                from the agricultural and NASA weather datasets.
                The selected indicator is shared with the weather
                analysis section above.
              </p>

              <div className="mt-6 grid gap-4 md:grid-cols-2">
                <StatCard
                  label="Matched Observations for Selected Indicator"
                  value={String(pairs.length)}
                />
                <StatCard
                  label="Pearson Correlation (r)"
                  value={
                    correlation === null
                      ? "Not defined"
                      : correlation.toFixed(3)
                  }
                />
              </div>

              <h4 className="mt-9 text-xl font-semibold">
                Corn Yield vs. {selectedMetric.label}
              </h4>

              <ScatterPlot
                pairs={pairs}
                weatherLabel={selectedMetric.label}
                unit={selectedMetric.unit}
              />

              <p className="mt-5 text-sm text-slate-400">
                Pearson correlation describes the linear association
                between two observed variables. It does not establish
                causation or predictive accuracy. Correlation is
                undefined when there are fewer than two valid pairs
                or either variable has zero variance.
              </p>

              <h4 className="mt-9 text-xl font-semibold">
                Matched County-Year Observations
              </h4>

              <div className="mt-5 overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-slate-800 text-slate-200">
                    <tr>
                      <th className="p-3">Year</th>
                      <th className="p-3">Yield (bu/acre)</th>
                      <th className="p-3">
                        {selectedMetric.label} ({selectedMetric.unit})
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {pairs.map((p) => (
                      <tr
                        key={p.year}
                        className="border-t border-slate-700"
                      >
                        <td className="p-3">{p.year}</td>
                        <td className="p-3">
                          {p.yieldValue.toFixed(2)}
                        </td>
                        <td className="p-3">
                          {p.weatherValue.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="mt-10 rounded-2xl border border-slate-800 bg-slate-900 p-6">
              <h3 className="text-2xl font-semibold">
                Historical Corn Yield Records — {county}
              </h3>

              <div className="mt-5 overflow-x-auto">
                <table className="w-full text-left">
                  <thead className="bg-slate-800">
                    <tr>
                      <th className="p-3">Year</th>
                      <th className="p-3">Yield (bu/acre)</th>
                      <th className="p-3">CV (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {countyYields.map((r) => (
                      <tr
                        key={r.Year}
                        className="border-t border-slate-700"
                      >
                        <td className="p-3">{r.Year}</td>
                        <td className="p-3">
                          {r.Value?.toFixed(1) ?? "N/A"}
                        </td>
                        <td className="p-3">
                          {r["CV (%)"] ?? "N/A"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <footer className="mt-12 border-t border-slate-800 pt-6 text-sm text-slate-400">
              SatHarvest | NASA POWER Weather Data and Missouri Corn Yield Analysis
            </footer>
          </>
        )}
      </div>
    </main>
  );
}
