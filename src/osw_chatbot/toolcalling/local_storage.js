// setLocalstorage
export function setLocalstorage(key, value) {
  localStorage
    .setItem
    (key, value);
}

// clearLocalstorage
export function clearLocalstorage() {
  localStorage.clear();
}

// getLocalstorage
export function getLocalstorage(key) {
  return localStorage.getItem(key);
}

// removeLocalstorage
export function removeLocalstorage(key) {
  localStorage.removeItem(key);
}

// getLocalstorageKeys
export function getLocalstorageKeys() {
  return Object.keys(localStorage);
}

// getLocalstorageValues
export function getLocalstorageValues() {
  return Object.values(localStorage);
}

// set example item to local storage
setLocalstorage("example", "example value");