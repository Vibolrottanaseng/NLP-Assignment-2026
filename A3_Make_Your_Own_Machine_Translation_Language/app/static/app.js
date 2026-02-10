const btn = document.getElementById("btn");
const input = document.getElementById("input");
const output = document.getElementById("output");
const statusEl = document.getElementById("status");
const attnEl = document.getElementById("attn");

async function translate() {
  const text = input.value.trim();
  if (!text) return;

  statusEl.textContent = "Translating...";
  output.textContent = "";
  attnEl.textContent = "";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await res.json();

    if (!data.ok) {
      statusEl.textContent = "Error ❌";
      output.textContent = data.error || "Unknown error";
      if (data.hint) attnEl.textContent = data.hint;
      return;
    }

    output.textContent = data.output || "(no output)";
    statusEl.textContent = "Done ✅";

    // show attention (debug)
    if (data.attention) {
      attnEl.textContent = JSON.stringify(
        { src_tokens: data.src_tokens, attention: data.attention },
        null,
        2
      );
    }
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Error ❌";
    output.textContent = "Could not reach /predict. Is the server running?";
  }
}

btn.addEventListener("click", translate);
