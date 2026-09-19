const CUBES = [
  { x: "7%", y: "14%", size: 56, delay: "-2s", duration: "14s" },
  { x: "19%", y: "69%", size: 34, delay: "-7s", duration: "17s" },
  { x: "32%", y: "8%", size: 45, delay: "-4s", duration: "16s" },
  { x: "48%", y: "77%", size: 64, delay: "-10s", duration: "20s" },
  { x: "63%", y: "19%", size: 39, delay: "-5s", duration: "15s" },
  { x: "76%", y: "61%", size: 52, delay: "-12s", duration: "19s" },
  { x: "88%", y: "23%", size: 30, delay: "-8s", duration: "13s" },
  { x: "92%", y: "82%", size: 46, delay: "-3s", duration: "18s" },
];

function SystemBackdrop() {
  return (
    <div className="system-backdrop" aria-hidden="true">
      <div className="system-fog fog-one" />
      <div className="system-fog fog-two" />
      <div className="system-grid" />

      {CUBES.map((cube, index) => (
        <div
          className={`system-cube cube-${index + 1}`}
          key={`${cube.x}-${cube.y}`}
          style={{
            "--cube-x": cube.x,
            "--cube-y": cube.y,
            "--cube-size": `${cube.size}px`,
            "--cube-delay": cube.delay,
            "--cube-duration": cube.duration,
          }}
        />
      ))}

      <span className="signal-light signal-pink" />
      <span className="signal-light signal-blue" />
      <span className="signal-light signal-green" />
      <div className="scanlines" />
    </div>
  );
}

export default SystemBackdrop;