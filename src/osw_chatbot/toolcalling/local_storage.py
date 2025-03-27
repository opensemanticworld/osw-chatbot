from pathlib import Path

import param
import panel as pn
from panel.models.esm import ESMEvent


# from panel.custom import ReactComponent
# from panel.custom import JSComponent
from panel.custom import AnyWidgetComponent

pn.extension()

local_storage_item_dict = {
    "key": "example",
    "value": "example value",
}


class LocalStorageWidget(AnyWidgetComponent):
    function_call = param.Dict()
    local_storage_item = param.Dict()

    _esm = """
    function render({ model, el }) {
        console.log("frontend loaded");
        model.on("change:function_call", () => {
            console.log("function_call", model.get("function_call"));
            localStorage.setItem(model.get("function_call")["key"], model.get("function_call")["value"]);
        });
        model.on("msg:custom", (e) => {
            console.log("local_storage_item", e);
            model.set("local_storage_item", localStorage.getItem(e["key"]));
            model.save_changes();
        });
      

    }
    export default { render };
    """  # noqa
    # model.on("change:local_storage_item", () => {
    #     console.log("local_storage_item", model.get("local_storage_item"));
    #     model.set("local_storage_item", localStorage.getItem(model.get("local_storage_item")["key"]));
    #     model.save_changes();
    # });

    @param.depends("function_call", watch=True)
    def _update_function_called(self):
        print("function_call", self.function_call)

    @param.depends("local_storage_item", watch=True)
    def _update_local_storage_item(self, event):
        print("local_storage_item", self.local_storage_item)


storage_widget = LocalStorageWidget()


set_button = pn.widgets.Button(
    name="Set LocalStorage Item", button_type="primary"
)
get_button = pn.widgets.Button(
    name="Get LocalStorage Item", button_type="primary"
)
button2 = pn.widgets.Button(name="Update Dictionary", button_type="primary")
key_input = pn.widgets.TextInput(name="Key", value="example")
value_input = pn.widgets.TextInput(
    name="Value", value="{'value': 'example_value'}"
)


def set_button_callback(event):
    # Increase the value of the type key by 1
    storage_widget.function_call = {
        "key": key_input.value,
        "value": value_input.value,
    }

    print("set button clicked")


def get_item_from_local_storage(key):
    """Get the value of the key"""
    return storage_widget._send_event(ESMEvent, data={"key": key})


def get_button_callback(event):
    """Get the value of the key"""

    result = get_item_from_local_storage(key=key_input.value)
    print(f"get button clicked: {result}")


set_button.on_click(callback=set_button_callback)
get_button.on_click(callback=get_button_callback)

app = pn.Column(
    pn.Row(storage_widget),  # invisible for functions
    pn.Row(key_input),
    pn.Row(value_input),
    pn.Row(set_button),
    pn.Row(get_button),
)
app.servable()

# # Works with jsx and css files
# class LocalStoragePanel(ReactComponent):

#     key = param.String()
#     value = param.String()
#     with open(Path(__file__).parent / "local_storage.jsx") as f:
#         _esm = f.read()
#     # _esm = """"""
#     _stylesheets = [Path(__file__).parent / "local_storage.css"]


# LocalStoragePanel().servable()


# class LocalStoragePanel(JSComponent):

#     key = param.String()
#     value = param.String()

#     # _esm = """
#     #     // setLocalstorage
#     #     export function setLocalstorage({ model }) {
#     #     localStorage
#     #         .setItem
#     #         (`${model.key}`, `${model.value}`);
#     #         model.on
#     #     }
#     # """

#     _esm = """
#     // setLocalstorage
#     export function setLocalstorage(key, value) {
#     localStorage
#         .setItem
#         (key, value);
#     }

#     // clearLocalstorage
#     export function clearLocalstorage() {
#     localStorage.clear();
#     }

#     // getLocalstorage
#     export function getLocalstorage(key) {
#     return localStorage.getItem(key);
#     }

#     // removeLocalstorage
#     export function removeLocalstorage(key) {
#     localStorage.removeItem(key);
#     }

#     // getLocalstorageKeys
#     export function getLocalstorageKeys() {
#     return Object.keys(localStorage);
#     }

#     // getLocalstorageValues
#     export function getLocalstorageValues() {
#     return Object.values(localStorage);
#     }

#     // set example item to local storage
#     setLocalstorage("example", "example value");
#     """


# LocalStoragePanel().servable()
