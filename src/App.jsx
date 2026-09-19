import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";
import AnnotationPanel from "./components/AnnotationPanel";

const GROUP_LABELS = {
  CN: "Cognitively Normal",
  MCI: "Mild Cognitive Impairment",
  AD_TRAJECTORY: "AD Trajectory",
};

const GROUP_COLORS = {
  CN: "#3b82f6",
  MCI: "#f59e0b",
  AD_TRAJECTORY: "#ef4444",
};

function thicknessColor(value) {
  const minimum = 1.5;
  const maximum = 3.5;
  const normalized = Math.max(
    0,
    Math.min(1, (value - minimum) / (maximum - minimum)),
  );

  const red = Math.round(235 - normalized * 180);
  const green = Math.round(90 + normalized * 100);
  const blue = Math.round(95 + normalized * 130);

  return `rgb(${red}, ${green}, ${blue})`;
}

function regionPosition(region, index) {
  const hemisphereDirection = region.hemisphereCode === "lh" ? -1 : 1;
  const localIndex = index % 34;

  const angle = (localIndex / 34) * Math.PI * 2;
  const verticalBand = ((localIndex % 7) - 3) / 3;

  return [
    hemisphereDirection * (1.15 + 0.42 * Math.cos(angle)),
    verticalBand * 0.82,
    0.92 * Math.sin(angle),
  ];
}

function BrainPrototype({
  regions,
  selectedGroup,
  selectedRegion,
  onSelectRegion,
}) {
  return (
    <>
      <ambientLight intensity={1.5} />
      <directionalLight position={[4, 6, 5]} intensity={2.2} />

      <group rotation={[0.05, 0, -0.04]}>
        <mesh position={[-0.78, 0, 0]} scale={[0.96, 1.28, 1.18]}>
          <sphereGeometry args={[1, 48, 48]} />
          <meshStandardMaterial
            color="#d7dde8"
            transparent
            opacity={0.2}
            roughness={0.7}
          />
        </mesh>

        <mesh position={[0.78, 0, 0]} scale={[0.96, 1.28, 1.18]}>
          <sphereGeometry args={[1, 48, 48]} />
          <meshStandardMaterial
            color="#d7dde8"
            transparent
            opacity={0.2}
            roughness={0.7}
          />
        </mesh>

        {regions.map((region, index) => {
          const value = region.groups[selectedGroup].mean;
          const isSelected = selectedRegion?.id === region.id;

          return (
            <mesh
              key={region.id}
              position={regionPosition(region, index)}
              scale={isSelected ? 1.5 : 1}
              onClick={(event) => {
                event.stopPropagation();
                onSelectRegion(region);
              }}
            >
              <sphereGeometry args={[0.115, 20, 20]} />
              <meshStandardMaterial
                color={isSelected ? "#ffffff" : thicknessColor(value)}
                emissive={isSelected ? "#7dd3fc" : "#000000"}
                emissiveIntensity={isSelected ? 0.75 : 0}
                roughness={0.35}
              />
            </mesh>
          );
        })}
      </group>

      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        minDistance={3.3}
        maxDistance={8}
      />
    </>
  );
}

function App() {
  const [viewerData, setViewerData] = useState(null);
  const [loadingError, setLoadingError] = useState("");
  const [selectedGroup, setSelectedGroup] = useState("CN");
  const [selectedRegion, setSelectedRegion] = useState(null);

  useEffect(() => {
    async function loadViewerData() {
      try {
        const response = await fetch("/data/cortical_summary.json");

        if (!response.ok) {
          throw new Error(`Data request failed: ${response.status}`);
        }

        const data = await response.json();
        setViewerData(data);
        setSelectedRegion(data.regions[0]);
      } catch (error) {
        setLoadingError(error.message);
      }
    }

    loadViewerData();
  }, []);

  const chartData = useMemo(() => {
    if (!selectedRegion) {
      return [];
    }

    return Object.entries(selectedRegion.groups).map(([group, values]) => ({
      group: GROUP_LABELS[group],
      shortGroup: group === "AD_TRAJECTORY" ? "AD Trajectory" : group,
      thickness: values.mean,
      participants: values.n,
      fill: GROUP_COLORS[group],
    }));
  }, [selectedRegion]);

  if (loadingError) {
    return (
      <main className="status-screen">
        <h1>Unable to load cortical data</h1>
        <p>{loadingError}</p>
      </main>
    );
  }

  if (!viewerData || !selectedRegion) {
    return (
      <main className="status-screen">
        <h1>Loading Cortical Explorer…</h1>
      </main>
    );
  }

  const metadata = viewerData.metadata;
  const currentMeasurement = selectedRegion.groups[selectedGroup];

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Interactive neuroimaging viewer</p>
          <h1>Cortical Explorer</h1>
          <p className="header-description">
            Explore regional cortical-thickness patterns across cognitive
            trajectories.
          </p>
        </div>

        <div className="cohort-summary">
          <span>{metadata.totalParticipants} participants</span>
          <span>{metadata.regionCount} cortical regions</span>
          <span>Thickness measured in {metadata.unit}</span>
        </div>
      </header>

      <section className="group-controls" aria-label="Select cognitive group">
        {metadata.groups.map((group) => (
          <button
            key={group}
            className={selectedGroup === group ? "group-button active" : "group-button"}
            onClick={() => setSelectedGroup(group)}
            type="button"
          >
            <span
              className="group-dot"
              style={{ backgroundColor: GROUP_COLORS[group] }}
            />
            <span>
              <strong>{GROUP_LABELS[group]}</strong>
              <small>{metadata.cohortCounts[group]} participants</small>
            </span>
          </button>
        ))}
      </section>

      <section className="viewer-grid">
        <article className="panel brain-panel">
          <div className="panel-heading">
            <div>
              <p className="section-label">3D cortical viewer</p>
              <h2>{GROUP_LABELS[selectedGroup]}</h2>
            </div>
            <p className="interaction-help">
              Drag to rotate · Scroll to zoom · Click a marker
            </p>
          </div>

          <div className="canvas-container">
            <Canvas camera={{ position: [0, 0, 5.2], fov: 42 }}>
              <BrainPrototype
                regions={viewerData.regions}
                selectedGroup={selectedGroup}
                selectedRegion={selectedRegion}
                onSelectRegion={setSelectedRegion}
              />
            </Canvas>
          </div>

          <div className="color-legend">
            <span>Thinner cortex</span>
            <div className="legend-gradient" />
            <span>Thicker cortex</span>
          </div>

          <p className="prototype-note">
            Current prototype: markers represent the 68 measured cortical
            regions. A matching anatomical surface will replace the translucent
            scaffold.
          </p>
        </article>

        <aside className="panel details-panel">
          <p className="section-label">Selected region</p>
          <h2>{selectedRegion.displayName}</h2>

          <div className="measurement-card">
            <span>Mean cortical thickness</span>
            <strong>{currentMeasurement.mean.toFixed(3)} mm</strong>
            <small>
              n = {currentMeasurement.n} · SD ={" "}
              {currentMeasurement.sd.toFixed(3)}
            </small>
          </div>

          <label className="region-selector">
            Browse all regions
            <select
              value={selectedRegion.id}
              onChange={(event) => {
                const nextRegion = viewerData.regions.find(
                  (region) => region.id === event.target.value,
                );
                setSelectedRegion(nextRegion);
              }}
            >
              {viewerData.regions.map((region) => (
                <option key={region.id} value={region.id}>
                  {region.displayName}
                </option>
              ))}
            </select>
          </label>

          <div className="chart-section">
            <h3>Group comparison</h3>
            <ResponsiveContainer width="100%" height={245}>
              <BarChart data={chartData} margin={{ top: 10, right: 5, left: -15 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dbe3ef" />
                <XAxis dataKey="shortGroup" tick={{ fontSize: 12 }} />
                <YAxis
                  domain={["dataMin - 0.15", "dataMax + 0.15"]}
                  tick={{ fontSize: 12 }}
                  unit=" mm"
                />
                <Tooltip
                  formatter={(value, name, item) => [
                    `${Number(value).toFixed(3)} mm (n=${item.payload.participants})`,
                    "Mean CT",
                  ]}
                />
                <Bar dataKey="thickness" radius={[7, 7, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="difference-list">
            <h3>Regional differences</h3>
            <p>
              <span>AD trajectory − CN</span>
              <strong>
                {selectedRegion.comparisons.AD_TRAJECTORY_minus_CN.toFixed(3)} mm
              </strong>
            </p>
            <p>
              <span>AD trajectory − MCI</span>
              <strong>
                {selectedRegion.comparisons.AD_TRAJECTORY_minus_MCI.toFixed(3)} mm
              </strong>
            </p>
            <p>
              <span>MCI − CN</span>
              <strong>
                {selectedRegion.comparisons.MCI_minus_CN.toFixed(3)} mm
              </strong>
            </p>
          </div>
          <AnnotationPanel
            selectedRegion={selectedRegion}
            selectedGroup={selectedGroup}
          />
        </aside>
      </section>
    </main>
  );
}

export default App;