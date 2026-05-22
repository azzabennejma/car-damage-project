import { FaCar } from "react-icons/fa";

function Navbar() {
  return (
    <div className="navbar">
      <h2><FaCar /> Car Damage Detection</h2>
      <div className="nav-buttons">
        <button>Dashboard</button>
        <button>History</button>
        <button>Model</button>
      </div>
    </div>
  );
}

export default Navbar;