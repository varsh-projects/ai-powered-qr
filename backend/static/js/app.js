/**
 * LinkShield AI — Frontend Application Logic
 * Integrates directly with real Flask API (/api/health, /api/scan).
 * No fake timers, no hardcoded predictions, safe DOM textContent manipulation.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const apiStatusDot = document.getElementById("apiStatusDot");
    const apiStatusText = document.getElementById("apiStatusText");
    const scanForm = document.getElementById("scanForm");
    const urlInput = document.getElementById("urlInput");
    const clearBtn = document.getElementById("clearBtn");
    const scanBtn = document.getElementById("scanBtn");
    const errorBanner = document.getElementById("errorBanner");
    const errorMessage = document.getElementById("errorMessage");
    
    // Results DOM Elements
    const resultsPanel = document.getElementById("resultsPanel");
    const resultCard = document.getElementById("resultCard");
    const verdictBadge = document.getElementById("verdictBadge");
    const verdictText = document.getElementById("verdictText");
    const scannedUrlDisplay = document.getElementById("scannedUrlDisplay");
    const riskMeterFill = document.getElementById("riskMeterFill");
    const riskMeterScore = document.getElementById("riskMeterScore");
    const indicatorsList = document.getElementById("indicatorsList");
    const featuresToggle = document.getElementById("featuresToggle");
    const featuresGrid = document.getElementById("featuresGrid");
    
    // Mobile navigation toggle
    const mobileMenuBtn = document.getElementById("mobileMenuBtn");
    const navLinks = document.getElementById("navLinks");

    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener("click", () => {
            navLinks.classList.toggle("mobile-open");
        });
    }

    // ===================================================================
    // 1. REAL API HEALTH CHECK
    // ===================================================================
    async function checkApiHealth() {
        try {
            const response = await fetch("/api/health", {
                headers: { "Accept": "application/json" },
                cache: "no-store"
            });

            if (response.ok) {
                const data = await response.json();
                if (data.model_loaded) {
                    apiStatusDot.className = "status-dot online";
                    apiStatusText.textContent = `API: Connected (${data.feature_count || 30} Features)`;
                } else {
                    apiStatusDot.className = "status-dot offline";
                    apiStatusText.textContent = "API: Model Degraded";
                }
            } else {
                apiStatusDot.className = "status-dot offline";
                apiStatusText.textContent = "API: Degraded (503)";
            }
        } catch (err) {
            apiStatusDot.className = "status-dot offline";
            apiStatusText.textContent = "API: Offline";
        }
    }

    // Initial health check on page load
    checkApiHealth();

    // ===================================================================
    // 2. INPUT INTERACTIONS (Clear button, Input validation)
    // ===================================================================
    function updateClearButton() {
        if (urlInput.value.length > 0) {
            clearBtn.classList.add("visible");
        } else {
            clearBtn.classList.remove("visible");
        }
    }

    urlInput.addEventListener("input", () => {
        updateClearButton();
        hideError();
    });

    clearBtn.addEventListener("click", () => {
        urlInput.value = "";
        updateClearButton();
        hideError();
        resultsPanel.classList.remove("visible");
        urlInput.focus();
    });

    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.classList.add("visible");
    }

    function hideError() {
        errorBanner.classList.remove("visible");
        errorMessage.textContent = "";
    }

    // ===================================================================
    // 3. SAMPLE BUTTONS (Demo Testing)
    // ===================================================================
    window.fillAndScan = function(sampleUrl) {
        urlInput.value = sampleUrl;
        updateClearButton();
        hideError();
        handleScanSubmit();
    };

    // ===================================================================
    // 4. SCAN SUBMISSION & BACKEND INTEGRATION
    // ===================================================================
    scanForm.addEventListener("submit", (e) => {
        e.preventDefault();
        handleScanSubmit();
    });

    async function handleScanSubmit() {
        const rawUrl = urlInput.value.trim();

        if (!rawUrl) {
            showError("Please enter a URL to analyze.");
            urlInput.focus();
            return;
        }

        hideError();
        setLoadingState(true);

        try {
            const response = await fetch("/api/scan", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                body: JSON.stringify({ url: rawUrl })
            });

            const data = await response.json();

            if (!response.ok) {
                const errorDetail = data.error || data.message || `Server returned status ${response.status}`;
                showError(errorDetail);
                resultsPanel.classList.remove("visible");
                return;
            }

            // Render the real model output
            renderScanResults(data);

        } catch (err) {
            showError("Unable to reach backend service. Please check that the Flask server is running.");
            resultsPanel.classList.remove("visible");
        } finally {
            setLoadingState(false);
        }
    }

    function setLoadingState(isLoading) {
        if (isLoading) {
            scanBtn.disabled = true;
            scanBtn.classList.add("loading");
            urlInput.disabled = true;
        } else {
            scanBtn.disabled = false;
            scanBtn.classList.remove("loading");
            urlInput.disabled = false;
        }
    }

    // ===================================================================
    // 5. RENDER SCAN RESULTS
    // ===================================================================
    function renderScanResults(data) {
        const isMalicious = data.label === "Malicious" || data.prediction === 1;
        const riskScore = typeof data.risk_score === "number" ? data.risk_score : 0;
        const maliciousProb = typeof data.malicious_probability === "number" 
            ? (data.malicious_probability * 100).toFixed(1) 
            : riskScore.toFixed(1);

        // Safe DOM insertion of scanned URL (Prevents XSS)
        scannedUrlDisplay.textContent = data.url || urlInput.value;

        // Apply theme styling
        resultCard.className = `result-card ${isMalicious ? "malicious" : "safe"}`;
        verdictBadge.className = `verdict-badge ${isMalicious ? "malicious" : "safe"}`;
        verdictText.textContent = isMalicious ? "MALICIOUS" : "SAFE";

        // Risk Meter
        riskMeterFill.className = `risk-meter-fill ${isMalicious ? "malicious-fill" : "safe-fill"}`;
        riskMeterFill.style.width = `${Math.min(100, Math.max(0, riskScore))}%`;
        
        riskMeterScore.className = `risk-meter-score ${isMalicious ? "malicious-score" : "safe-score"}`;
        riskMeterScore.textContent = `${riskScore}% (Probability: ${maliciousProb}%)`;

        // Render Risk Indicators
        indicatorsList.innerHTML = "";
        const indicators = Array.isArray(data.risk_indicators) ? data.risk_indicators : [];

        if (indicators.length === 0) {
            const li = document.createElement("li");
            li.className = "indicator-item clean";
            li.innerHTML = `
                <svg class="indicator-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                </svg>
                <span>No prominent structural or lexical threat indicators detected.</span>
            `;
            indicatorsList.appendChild(li);
        } else {
            indicators.forEach(indicatorText => {
                const li = document.createElement("li");
                const isThreat = isMalicious || !indicatorText.toLowerCase().includes("no prominent");
                li.className = `indicator-item ${isThreat ? "flagged" : "clean"}`;

                const iconSvg = isThreat
                    ? `<svg class="indicator-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                       </svg>`
                    : `<svg class="indicator-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                       </svg>`;

                const span = document.createElement("span");
                span.textContent = indicatorText;

                li.innerHTML = iconSvg;
                li.appendChild(span);
                indicatorsList.appendChild(li);
            });
        }

        // Render Feature Breakdown Grid
        renderFeatureGrid(data.features || {});

        // Reveal panel and scroll gently into view
        resultsPanel.classList.add("visible");
        resultsPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function renderFeatureGrid(features) {
        featuresGrid.innerHTML = "";
        const keys = Object.keys(features);

        if (keys.length === 0) {
            featuresGrid.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem;">Feature breakdown unavailable.</div>`;
            return;
        }

        // Sort keys logically
        keys.sort().forEach(key => {
            const val = features[key];
            const badge = document.createElement("div");
            badge.className = "feature-badge";

            const nameSpan = document.createElement("span");
            nameSpan.className = "feature-name";
            nameSpan.textContent = key;

            const valSpan = document.createElement("span");
            valSpan.className = "feature-val";
            valSpan.textContent = val;

            badge.appendChild(nameSpan);
            badge.appendChild(valSpan);
            featuresGrid.appendChild(badge);
        });
    }

    // Toggle features accordion
    if (featuresToggle && featuresGrid) {
        featuresToggle.addEventListener("click", () => {
            const isOpen = featuresGrid.classList.contains("open");
            if (isOpen) {
                featuresGrid.classList.remove("open");
                featuresToggle.classList.remove("open");
            } else {
                featuresGrid.classList.add("open");
                featuresToggle.classList.add("open");
            }
        });
    }
});
