export function render({ model }) {
  const [value, setValue] = model.useState("value");
  return (
    <button onClick={e => setValue(value+1)}>
      count is {value}
    </button>
  );
}
