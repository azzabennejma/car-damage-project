export default function Sidebar({ role }) {

  const userMenu = [
    "Dashboard",
    "Inference",
    "History"
  ];

  const developerMenu = [
    "Dashboard",
    "Inference",
    "Training",
    "Dataset",
    "MLflow",
    "Pipeline",
    "History",
    "Models",
    "Settings"
  ];

  const items =
    role === "developer"
      ? developerMenu
      : userMenu;

  return (
    <div className="sidebar">

      {items.map(item=>(
        <button key={item}>
          {item}
        </button>
      ))}

    </div>
  );
}