import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowDown,
  ArrowUpRight,
  Binary,
  Braces,
  Check,
  ChevronRight,
  CircleDot,
  Code2,
  Cpu,
  Github,
  Radio,
  ScanSearch,
  Sparkles,
  Waves,
  Zap,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";

type ResultPoint = {
  snr: number;
  theory: number;
  uncoded: number;
  ldpc: number;
  ldpcLabel: string;
  fer: string;
  note: string;
};

const results: ResultPoint[] = [
  {
    snr: 0,
    theory: 7.865,
    uncoded: 8.44,
    ldpc: 19.477,
    ldpcLabel: "19.477%",
    fer: "99.33%",
    note: "Below the short code’s decoding threshold.",
  },
  {
    snr: 2,
    theory: 3.751,
    uncoded: 4.146,
    ldpc: 4.792,
    ldpcLabel: "4.792%",
    fer: "32.00%",
    note: "In the waterfall: 68% of frames converge.",
  },
  {
    snr: 4,
    theory: 1.25,
    uncoded: 1.359,
    ldpc: 0.0023,
    ldpcLabel: "0 / 21,600 bits",
    fer: "0 / 150",
    note: "All frames converge in 3.1 iterations on average.",
  },
  {
    snr: 6,
    theory: 0.2388,
    uncoded: 0.2824,
    ldpc: 0.0023,
    ldpcLabel: "0 / 21,600 bits",
    fer: "0 / 150",
    note: "Zero observed coded bit or frame errors.",
  },
  {
    snr: 8,
    theory: 0.0191,
    uncoded: 0.0255,
    ldpc: 0.0023,
    ldpcLabel: "0 / 21,600 bits",
    fer: "0 / 150",
    note: "Measured across the same joint impairments.",
  },
];

const pipeline = [
  {
    id: "acquire",
    eyebrow: "01 · FIND",
    name: "Burst acquisition",
    icon: ScanSearch,
    detail:
      "Normalized preamble correlation searches carrier frequency, sample phase, and frame lag without assuming where the burst begins.",
    metric: "±5% pull-in range",
  },
  {
    id: "align",
    eyebrow: "02 · ALIGN",
    name: "Timing + carrier",
    icon: Waves,
    detail:
      "A bounded ML tone search estimates residual CFO. Complex least squares then removes phase and gain without fragile phase unwrapping.",
    metric: "0.31-sample offset",
  },
  {
    id: "decide",
    eyebrow: "03 · DECIDE",
    name: "Soft demapping",
    icon: Binary,
    detail:
      "Exact AWGN log-likelihood ratios carry confidence into decoding instead of throwing information away with hard symbol decisions.",
    metric: "Gray QPSK",
  },
  {
    id: "decode",
    eyebrow: "04 · RECOVER",
    name: "LDPC decoding",
    icon: Cpu,
    detail:
      "A deterministic sparse Tanner graph and normalized min-sum decoder recover 144 information bits with syndrome-based early stopping.",
    metric: "(288, 144) code",
  },
];

const problems = [
  {
    number: "01",
    label: "Estimator bias",
    title: "CFO at the wrong sampling instant",
    copy: "The fourth-power estimator looked confident but was biased between RRC symbol decisions. Acquisition moved to matched, symbol-spaced hypotheses.",
  },
  {
    number: "02",
    label: "Hidden variable",
    title: "Gain changed the effective Eb/N0",
    copy: "Signal gain originally moved without the noise. The channel now scales both together before AGC, keeping Eb/N0 as the single SNR control.",
  },
  {
    number: "03",
    label: "Failure semantics",
    title: "A failed decoder added errors",
    copy: "Non-converged min-sum output can oscillate. Failure is now explicit and falls back to channel decisions instead of silently making BER worse.",
  },
];

function SignalField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    let animationFrame = 0;
    let width = 0;
    let height = 0;
    let ratio = 1;
    const pointer = { x: 0.64, y: 0.44 };
    const particles = Array.from({ length: 46 }, (_, index) => ({
      phase: index * 1.71,
      orbit: 0.16 + ((index * 37) % 100) / 190,
      speed: 0.00018 + ((index * 13) % 12) * 0.000015,
      size: 0.7 + ((index * 29) % 10) / 7,
    }));

    const resize = () => {
      const bounds = canvas.getBoundingClientRect();
      ratio = Math.min(window.devicePixelRatio || 1, 2);
      width = bounds.width;
      height = bounds.height;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };

    const onPointerMove = (event: PointerEvent) => {
      const bounds = canvas.getBoundingClientRect();
      pointer.x = (event.clientX - bounds.left) / bounds.width;
      pointer.y = (event.clientY - bounds.top) / bounds.height;
    };

    const render = (time: number) => {
      context.clearRect(0, 0, width, height);
      const centerX = width * (0.5 + (pointer.x - 0.5) * 0.09);
      const centerY = height * (0.47 + (pointer.y - 0.5) * 0.06);

      context.strokeStyle = "rgba(71, 207, 255, 0.09)";
      context.lineWidth = 1;
      for (let x = 20; x < width; x += 42) {
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
      }
      for (let y = 20; y < height; y += 42) {
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
      }

      const points = particles.map((particle) => {
        const angle =
          particle.phase + (reducedMotion ? 0 : time * particle.speed);
        const radius = Math.min(width, height) * particle.orbit;
        return {
          x: centerX + Math.cos(angle) * radius * 1.32,
          y: centerY + Math.sin(angle) * radius * 0.72,
          size: particle.size,
        };
      });

      points.forEach((point, index) => {
        points.slice(index + 1).forEach((other) => {
          const distance = Math.hypot(point.x - other.x, point.y - other.y);
          if (distance < 82) {
            context.beginPath();
            context.moveTo(point.x, point.y);
            context.lineTo(other.x, other.y);
            context.strokeStyle = `rgba(66, 215, 255, ${0.11 * (1 - distance / 82)})`;
            context.stroke();
          }
        });
      });

      points.forEach((point, index) => {
        const isAmber = index % 11 === 0;
        context.beginPath();
        context.arc(point.x, point.y, point.size, 0, Math.PI * 2);
        context.fillStyle = isAmber
          ? "rgba(255, 177, 78, 0.85)"
          : "rgba(128, 234, 255, 0.8)";
        context.fill();
      });

      const qpsk = [
        [-1, -1],
        [1, -1],
        [-1, 1],
        [1, 1],
      ];
      qpsk.forEach(([x, y], index) => {
        const pulse = reducedMotion ? 1 : 1 + Math.sin(time * 0.003 + index) * 0.15;
        const qx = centerX + x * Math.min(width, height) * 0.18;
        const qy = centerY + y * Math.min(width, height) * 0.18;
        const glow = context.createRadialGradient(qx, qy, 0, qx, qy, 25 * pulse);
        glow.addColorStop(0, "rgba(117, 232, 255, 0.9)");
        glow.addColorStop(0.25, "rgba(49, 199, 255, 0.28)");
        glow.addColorStop(1, "rgba(49, 199, 255, 0)");
        context.fillStyle = glow;
        context.fillRect(qx - 30, qy - 30, 60, 60);
        context.beginPath();
        context.arc(qx, qy, 3.2, 0, Math.PI * 2);
        context.fillStyle = "#dffaff";
        context.fill();
      });

      if (!reducedMotion) animationFrame = requestAnimationFrame(render);
    };

    resize();
    render(0);
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    canvas.addEventListener("pointermove", onPointerMove);

    return () => {
      observer.disconnect();
      canvas.removeEventListener("pointermove", onPointerMove);
      cancelAnimationFrame(animationFrame);
    };
  }, [reducedMotion]);

  return <canvas className="signal-canvas" ref={canvasRef} aria-hidden="true" />;
}

function SignalMonitor() {
  const reducedMotion = useReducedMotion();
  const path =
    "M0 66 C16 65 25 63 36 66 S56 76 67 64 S83 37 97 68 S112 107 127 60 S147 17 164 67 S184 118 201 62 S220 24 237 66 S256 91 270 64 S291 55 304 66 S326 68 340 65";

  return (
    <div className="signal-monitor" aria-label="Animated signal monitor">
      <div className="monitor-topline">
        <span>
          <i className="status-dot" /> LIVE SIGNAL
        </span>
        <span>2 SPS · RRC α 0.35</span>
      </div>
      <SignalField />
      <svg className="waveform" viewBox="0 0 340 132" role="img" aria-label="QPSK waveform">
        <defs>
          <linearGradient id="wave" x1="0" x2="1">
            <stop offset="0" stopColor="#31c7ff" stopOpacity="0" />
            <stop offset="0.35" stopColor="#77e8ff" />
            <stop offset="0.72" stopColor="#ffb14e" />
            <stop offset="1" stopColor="#31c7ff" stopOpacity="0" />
          </linearGradient>
        </defs>
        <motion.path
          d={path}
          fill="none"
          stroke="url(#wave)"
          strokeWidth="1.8"
          initial={{ pathLength: 0, opacity: 0 }}
          animate={{ pathLength: 1, opacity: 1 }}
          transition={{ duration: reducedMotion ? 0 : 2.2, ease: "easeInOut" }}
        />
      </svg>
      <div className="monitor-label label-cfo">
        <span>CFO</span>
        <strong>+1.30%</strong>
      </div>
      <div className="monitor-label label-lock">
        <span>LOCK</span>
        <strong>ACQUIRED</strong>
      </div>
      <div className="monitor-footer">
        <span>I</span>
        <span className="meter">
          <i />
        </span>
        <span>Q</span>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  suffix,
  detail,
  delay,
}: {
  label: string;
  value: string;
  suffix?: string;
  detail: string;
  delay: number;
}) {
  return (
    <motion.article
      className="metric-card glow-card"
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.5 }}
      transition={{ delay, duration: 0.5 }}
      whileHover={{ y: -5 }}
    >
      <span>{label}</span>
      <div>
        <strong>{value}</strong>
        {suffix && <em>{suffix}</em>}
      </div>
      <p>{detail}</p>
    </motion.article>
  );
}

function Pipeline() {
  const [active, setActive] = useState(0);
  const ActiveIcon = pipeline[active].icon;

  return (
    <section className="section pipeline-section" id="architecture">
      <div className="section-heading">
        <div>
          <span className="section-index">01 / SIGNAL CHAIN</span>
          <h2>A receiver that earns its lock.</h2>
        </div>
        <p>
          Every stage handles a real uncertainty. Select a block to inspect why
          it exists.
        </p>
      </div>

      <div className="pipeline-shell">
        <div className="pipeline-rail" role="tablist" aria-label="Receiver stages">
          {pipeline.map((stage, index) => {
            const Icon = stage.icon;
            return (
              <button
                key={stage.id}
                role="tab"
                aria-selected={active === index}
                className={active === index ? "pipeline-button active" : "pipeline-button"}
                onClick={() => setActive(index)}
              >
                <span className="pipeline-node">
                  <Icon size={18} />
                </span>
                <span>
                  <small>{stage.eyebrow}</small>
                  <strong>{stage.name}</strong>
                </span>
                {index < pipeline.length - 1 && <i className="pipeline-line" />}
              </button>
            );
          })}
        </div>

        <div className="stage-detail">
          <div className="stage-radar" aria-hidden="true">
            <span />
            <motion.i
              key={active}
              initial={{ rotate: -80 }}
              animate={{ rotate: 280 }}
              transition={{ duration: 1.25, ease: "easeOut" }}
            />
            <ActiveIcon size={28} />
          </div>
          <AnimatePresence mode="wait">
            <motion.div
              key={pipeline[active].id}
              initial={{ opacity: 0, x: 12 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.28 }}
            >
              <span className="stage-kicker">{pipeline[active].eyebrow}</span>
              <h3>{pipeline[active].name}</h3>
              <p>{pipeline[active].detail}</p>
              <div className="stage-metric">
                <Activity size={15} />
                {pipeline[active].metric}
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

function BerExplorer() {
  const [pointIndex, setPointIndex] = useState(2);
  const [mode, setMode] = useState<"uncoded" | "ldpc">("ldpc");
  const selected = results[pointIndex];

  const chart = useMemo(() => {
    const width = 560;
    const height = 260;
    const padX = 34;
    const padY = 26;
    const values = results.map((item) =>
      mode === "uncoded" ? item.uncoded : item.ldpc,
    );
    const y = (value: number) => {
      const log = Math.log10(Math.max(value, 0.001));
      return padY + ((1.35 - log) / (1.35 - -3.2)) * (height - padY * 2);
    };
    const points = values.map((value, index) => ({
      x: padX + (index / (values.length - 1)) * (width - padX * 2),
      y: y(value),
    }));
    return {
      width,
      height,
      points,
      path: points.map((point, index) => `${index ? "L" : "M"} ${point.x} ${point.y}`).join(" "),
      area: `${points.map((point, index) => `${index ? "L" : "M"} ${point.x} ${point.y}`).join(" ")} L ${points.at(-1)!.x} ${height - padY} L ${points[0].x} ${height - padY} Z`,
    };
  }, [mode]);

  return (
    <section className="section ber-section" id="results">
      <div className="section-heading">
        <div>
          <span className="section-index">02 / MEASURED RESULTS</span>
          <h2>Find the waterfall.</h2>
        </div>
        <p>
          150 frames per point with CFO, phase error, fractional timing, gain,
          and AWGN all active together.
        </p>
      </div>

      <div className="ber-grid">
        <div className="chart-card">
          <div className="chart-toolbar">
            <div>
              <span className="chart-live">
                <i /> MEASURED BER
              </span>
              <strong>{mode === "ldpc" ? "LDPC decoded" : "Full receiver"}</strong>
            </div>
            <div className="mode-toggle" aria-label="BER series">
              <button
                className={mode === "uncoded" ? "active" : ""}
                onClick={() => setMode("uncoded")}
              >
                Receiver
              </button>
              <button
                className={mode === "ldpc" ? "active" : ""}
                onClick={() => setMode("ldpc")}
              >
                LDPC
              </button>
            </div>
          </div>

          <div className="chart-wrap">
            <svg
              viewBox={`0 0 ${chart.width} ${chart.height}`}
              role="img"
              aria-label={`${mode} bit error rate by Eb/N0`}
            >
              {[52, 100, 148, 196, 234].map((line) => (
                <line
                  key={line}
                  x1="34"
                  x2="526"
                  y1={line}
                  y2={line}
                  className="chart-gridline"
                />
              ))}
              <defs>
                <linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0" stopColor="#31c7ff" stopOpacity=".32" />
                  <stop offset="1" stopColor="#31c7ff" stopOpacity="0" />
                </linearGradient>
                <filter id="line-glow">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <motion.path
                key={`${mode}-area`}
                d={chart.area}
                fill="url(#chart-fill)"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              />
              <motion.path
                key={mode}
                d={chart.path}
                fill="none"
                stroke="#77e8ff"
                strokeWidth="2.5"
                filter="url(#line-glow)"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              />
              {chart.points.map((point, index) => (
                <g key={results[index].snr}>
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={pointIndex === index ? 8 : 4}
                    className={pointIndex === index ? "chart-point active" : "chart-point"}
                    onClick={() => setPointIndex(index)}
                  />
                  <text x={point.x} y="253" textAnchor="middle" className="chart-label">
                    {results[index].snr} dB
                  </text>
                </g>
              ))}
            </svg>
            <span className="axis-label">BIT ERROR RATE · LOG SCALE</span>
          </div>

          <label className="snr-control">
            <span>EB/N0 TEST POINT</span>
            <input
              type="range"
              min="0"
              max="4"
              step="1"
              value={pointIndex}
              onChange={(event) => setPointIndex(Number(event.target.value))}
              aria-label="Select Eb/N0 test point"
            />
            <output>{selected.snr} dB</output>
          </label>
        </div>

        <aside className="result-readout">
          <div className="readout-top">
            <span>TEST POINT</span>
            <strong>{selected.snr}<small>dB</small></strong>
          </div>
          <div className="readout-row">
            <span>Full receiver BER</span>
            <strong>{selected.uncoded}%</strong>
          </div>
          <div className="readout-row">
            <span>LDPC BER</span>
            <strong>{selected.ldpcLabel}</strong>
          </div>
          <div className="readout-row">
            <span>LDPC FER</span>
            <strong>{selected.fer}</strong>
          </div>
          <div className="readout-verdict">
            <Zap size={18} />
            <p>{selected.note}</p>
          </div>
          <p className="readout-footnote">
            Zero-error points use a plotting bound only for the log chart. The
            source CSV preserves measured zero.
          </p>
        </aside>
      </div>
    </section>
  );
}

function App() {
  const reducedMotion = useReducedMotion();

  return (
    <div className="site-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="SDR Receiver home">
          <span className="brand-mark">
            <Radio size={18} />
          </span>
          <span>
            SDR<span>/</span>RX
          </span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#architecture">Architecture</a>
          <a href="#results">Results</a>
          <a href="#decisions">Decisions</a>
        </nav>
        <a
          className="github-link"
          href="https://github.com/asp53826/sdr-receiver"
          target="_blank"
          rel="noreferrer"
        >
          <Github size={17} />
          <span>View source</span>
          <ArrowUpRight size={15} />
        </a>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <motion.div
              className="eyebrow"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <span className="eyebrow-pulse" />
              COMPLETE QPSK BURST RECEIVER
              <span className="eyebrow-code">BUILD 01</span>
            </motion.div>
            <motion.h1
              initial={{ opacity: 0, y: 28 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: reducedMotion ? 0 : 0.12, duration: 0.65 }}
            >
              The signal arrives
              <br />
              <span>imperfect.</span>
              <br />
              The bits don’t.
            </motion.h1>
            <motion.p
              className="hero-lede"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: reducedMotion ? 0 : 0.24, duration: 0.58 }}
            >
              From waveform to recovered information: pulse shaping, joint
              synchronization, soft decisions, and sparse LDPC decoding under
              five simultaneous channel impairments.
            </motion.p>
            <motion.div
              className="hero-actions"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: reducedMotion ? 0 : 0.38 }}
            >
              <a className="primary-button" href="#architecture">
                Explore the receiver
                <ArrowDown size={17} />
              </a>
              <a
                className="text-button"
                href="https://github.com/asp53826/sdr-receiver"
                target="_blank"
                rel="noreferrer"
              >
                <Code2 size={17} />
                Inspect the code
              </a>
            </motion.div>
          </div>

          <motion.div
            className="hero-instrument"
            initial={{ opacity: 0, scale: 0.96, x: 22 }}
            animate={{ opacity: 1, scale: 1, x: 0 }}
            transition={{ delay: reducedMotion ? 0 : 0.18, duration: 0.8 }}
          >
            <div className="instrument-chrome">
              <div>
                <span>RX / ACQUISITION</span>
                <small>CH 01</small>
              </div>
              <div className="chrome-lights">
                <i />
                <i />
                <i />
              </div>
            </div>
            <SignalMonitor />
            <div className="instrument-specs">
              <div>
                <span>MOD</span>
                <strong>QPSK</strong>
              </div>
              <div>
                <span>CODE</span>
                <strong>R 1/2</strong>
              </div>
              <div>
                <span>SYNC</span>
                <strong>LOCKED</strong>
              </div>
            </div>
          </motion.div>
        </section>

        <section className="metric-strip" aria-label="Project metrics">
          <MetricCard
            label="REGRESSION SUITE"
            value="20"
            suffix="tests"
            detail="Algebraic, synchronization, and full-chain coverage."
            delay={0}
          />
          <MetricCard
            label="CODE STRUCTURE"
            value="288"
            suffix="/ 144"
            detail="Deterministic regular LDPC Tanner graph."
            delay={0.08}
          />
          <MetricCard
            label="CLEAN RECOVERY"
            value="0"
            suffix="errors"
            detail="Observed across 21,600 coded bits at 4 dB."
            delay={0.16}
          />
          <MetricCard
            label="JOINT IMPAIRMENTS"
            value="5"
            suffix="active"
            detail="CFO, phase, timing, gain, and AWGN."
            delay={0.24}
          />
        </section>

        <Pipeline />
        <BerExplorer />

        <section className="section decisions-section" id="decisions">
          <div className="section-heading">
            <div>
              <span className="section-index">03 / ENGINEERING DECISIONS</span>
              <h2>The failures shaped the receiver.</h2>
            </div>
            <p>
              The strongest parts of the implementation came from defects that
              looked correct until measurement proved otherwise.
            </p>
          </div>
          <div className="decision-grid">
            {problems.map((problem, index) => (
              <motion.article
                key={problem.number}
                className="decision-card"
                initial={{ opacity: 0, y: 25 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.3 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="decision-topline">
                  <span>{problem.number}</span>
                  <small>{problem.label}</small>
                </div>
                <h3>{problem.title}</h3>
                <p>{problem.copy}</p>
                <div className="decision-resolved">
                  <Check size={14} />
                  Regression covered
                </div>
              </motion.article>
            ))}
          </div>
        </section>

        <section className="cta-section">
          <div className="cta-scan" aria-hidden="true" />
          <div>
            <span className="section-index">SOURCE / REPRODUCIBLE</span>
            <h2>Run the entire signal chain.</h2>
            <p>
              The benchmark, raw measurements, tests, and implementation are
              public. No simulated dashboard data—just the receiver’s real
              output.
            </p>
          </div>
          <a
            href="https://github.com/asp53826/sdr-receiver"
            target="_blank"
            rel="noreferrer"
            className="cta-button"
          >
            <Github size={19} />
            Open repository
            <ChevronRight size={17} />
          </a>
        </section>
      </main>

      <footer>
        <div className="footer-brand">
          <CircleDot size={16} />
          <span>DESIGNED AS AN ENGINEERING EXHIBIT</span>
        </div>
        <p>Python · NumPy · SciPy · Signal processing · Coding theory</p>
        <a href="#top">
          BACK TO TOP
          <ArrowUpRight size={13} />
        </a>
      </footer>

      <div className="corner-readout" aria-hidden="true">
        <Braces size={14} />
        <span>RX_READY</span>
        <Sparkles size={12} />
      </div>
    </div>
  );
}

export default App;
