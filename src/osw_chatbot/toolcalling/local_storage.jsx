
export function render({ model }) {
  const [key, setKey] = model.useState("key");
  const [value, setValue] = model.useState("value");


  // Handle save to local storage
  const saveToLocalStorage = () => {
    localStorage.setItem
    (key, value);
  }
  // New form data component
  
  return (
    <div>
      <h1>Local Storage</h1>
      <label>Key:</label>
      <input 
        type="text" 
        name="key"
        value={key} 
        onChange={e => setKey(e.target.value)} 
        style={{marginRight: "10px"}}
      />
      <label>Value:</label>
      <input 
        type="text" 
        name="value"
        value={value} 
        onChange={e => setValue(e.target.value)} 
        style={{marginRight: "10px"}}
      />
      <button onClick={saveToLocalStorage}>Save</button>
      <button onClick={() => localStorage.clear()}>Clear</button>
    </div>

  );
}
