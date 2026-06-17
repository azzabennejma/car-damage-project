export default function StatCard({
 title,
 value,
 subtitle
}) {

 return (
   <div className="card">

      <h4>{title}</h4>

      <h2>{value}</h2>

      <small>{subtitle}</small>

   </div>
 );
}