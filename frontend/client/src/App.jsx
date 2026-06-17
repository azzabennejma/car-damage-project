import { Routes, Route } from "react-router-dom";
import Login from "./login";
import Admin from "./admin";
import User from "./user";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route path="/admin" element={<Admin />} />
      <Route path="/user" element={<User />} />
    </Routes>
  );
}

export default App;