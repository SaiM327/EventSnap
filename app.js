// Shared EventSnap frontend helpers
const API_BASE = "http://localhost:8001";

function getToken() {
  return localStorage.getItem("eventsnap_token");
}

function setSession(token, user) {
  localStorage.setItem("eventsnap_token", token);
  localStorage.setItem("eventsnap_user", JSON.stringify(user));
}

function clearSession() {
  localStorage.removeItem("eventsnap_token");
  localStorage.removeItem("eventsnap_user");
}

// Redirect to login if not authenticated. Call at the top of protected pages.
function requireAuth() {
  if (!getToken()) {
    window.location.href = "login.html";
  }
}

// fetch() wrapper that attaches the auth token and bounces to login on 401.
async function authFetch(path, options = {}) {
  const headers = options.headers ? { ...options.headers } : {};
  headers["Authorization"] = `Bearer ${getToken()}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (res.status === 401) {
    clearSession();
    window.location.href = "login.html";
    throw new Error("Not authenticated");
  }
  return res;
}

async function logout() {
  try {
    await authFetch("/auth/logout", { method: "POST" });
  } catch {
    // already redirected or backend down; clear locally either way
  }
  clearSession();
  window.location.href = "login.html";
}
