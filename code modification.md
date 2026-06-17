# CarDamageDetection.jsx — KPI Cards Update

## STEP 1 — Find this block

```jsx
return (
  <div style={{
      display: "grid",
      gridTemplateColumns: "260px 1fr 260px",
```

## STEP 2 — DELETE EVERYTHING until the end of the 3-column layout

---

## STEP 3 — REPLACE WITH THIS

```jsx
  <div
    style={{
      background: "#0d1117",
      color: "white",
      minHeight: "100vh",
      padding: "20px"
    }}
  >
```

---

## KPI ROW (ADD THIS)

```jsx
{/* KPI ROW */}
<div
  style={{
    display: "grid",
    gridTemplateColumns: "repeat(4,1fr)",
    gap: "15px",
    marginBottom: "20px"
  }}
>
```

---

## KPI CARDS

```jsx
<div style={card}>
  <h4>Dataset Images</h4>
  <h1>{dataset?.uploaded || 0}</h1>
</div>

<div style={card}>
  <h4>Current Model</h4>
  <h1>v12</h1>
</div>

<div style={card}>
  <h4>mAP50</h4>
  <h1>{mlflow?.map50 || 0}%</h1>
</div>

<div style={card}>
  <h4>Pipeline</h4>
  <h1>Healthy</h1>
</div>
```

---

## STEP 4 — CLOSE DIV

```jsx
</div>
```