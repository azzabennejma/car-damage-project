import { useState } from "react";
import "./App.css";

function Login() {
  const [isLogin, setIsLogin] = useState(false);

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!isLogin && password !== confirmPassword) {
    alert("Passwords do not match!");
    return;
    }
    // =========================================
    // 🧪 DEBUG MODE (NO BACKEND / NO DATABASE)
    // =========================================

    /*if (isLogin) {
      // simulate fake authentication delay
      setTimeout(() => {
        alert("Login successful ");

        localStorage.setItem("user", username);

        // 🔥 DEBUG ROLE LOGIC:
        // change this manually for testing
        const role = username === "admin" ? "user" : "admin";

        localStorage.setItem("role", role);

        if (role === "user") {
          window.location.href = "/user";
        } else {
          window.location.href = "/admin";
        }
      }, 500);

      return;
    }

    // REGISTER MODE (DEBUG)
    if (password !== confirmPassword) {
      alert("Passwords do not match!");
      return;
    }*/

    try {

  const endpoint = isLogin
    ? "http://localhost:8000/login"
    : "http://localhost:8000/register";

  const payload = isLogin
    ? {
        username,
        password
      }
    : {
        username,
        email,
        password
      };

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  
  const data = await res.json();
  localStorage.setItem("username", data.username);
  
  if (!res.ok) {
    alert(data.detail);
    return;
  }

  if (isLogin) {

    localStorage.setItem(
      "user",
      data.username
    );

    localStorage.setItem(
      "role",
      data.role
    );

    if (data.role === "admin") {
      window.location.href = "/admin";
    } else {
      window.location.href = "/user";
    }

  } else {

    alert("Account created successfully");
    setIsLogin(true);

  }

  } catch (err) {
    alert("Server error");
  }
    /*setTimeout(() => {
      alert("Account created (DEBUG MODE)");
      setIsLogin(true);
    }, 500);
    */
    return;

    /* =========================================
       🚨 BACKEND (DISABLED FOR NOW)

    try {
      const res = await fetch("http://localhost:8000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await res.json();

      if (res.ok) {
        localStorage.setItem("user", data.username);
        localStorage.setItem("role", data.role);

        if (data.role === "admin") {
          window.location.href = "/admin";
        } else {
          window.location.href = "/user";
        }
      }
    } catch (err) {
      alert("Server error");
    }

    ========================================= */
  };

  return (
    <div className="login-container">
      <div className="login-card">


        <p 
          style={{
          fontSize: "20px",
          fontWeight: "700",
          color: "white",
          marginBottom: "20px",
          }}
        >
  {isLogin ? "LOG IN" : "WELCOME"}
</p>

        <form onSubmit={handleSubmit}>

          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) =>
              setUsername(e.target.value)
            }
            required
          />

          {!isLogin && (
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) =>
                setEmail(e.target.value)
              }
              required
            />
          )}

          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) =>
              setPassword(e.target.value)
            }
            required
          />
         

          {!isLogin && (
            <input
              type="password"
              placeholder="Confirm Password"
              value={confirmPassword}
              onChange={(e) =>
                setConfirmPassword(e.target.value)
              }
              required
            />
          )}

          <button type="submit">
            {isLogin ? "Log in" : "Sign Up"}
          </button>
        </form>

        <p className="switch-text">
          {isLogin ? (
            <>
              Don't have an account?{" "}
              <span onClick={() => setIsLogin(false)}>
                Sign up
              </span>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <span onClick={() => setIsLogin(true)}>
                Login
              </span>
            </>
          )}
        </p>

      </div>
      <img
        src="/seca.png"
        alt="Enterprise Logo"
        className="enterprise-logo"
      />
    </div>
  );
}

export default Login;