import panel as pn
import param
from panel.custom import AnyWidgetComponent

pn.extension()


class ChatFrontendWidget(AnyWidgetComponent):
    function_call = param.Dict()
    function_called = param.Dict()

    _esm = """
    function render({ model, el }) {
      console.log("frontend loaded");
      model.on("change:function_call", () => {
        console.log("function_call", model.get("function_call"));
        //model.set("function_called", {"name": "test_function", "args": [1, 2, 3]});
        window.parent.postMessage(model.get("function_call"), "*");
        //model.save_changes();
      });
      window.addEventListener("message", (event) => {
        console.log("Received data from parant iframe ", event.origin, event.data);
        if (event.data["type"] === "function_call_result") {
          model.set("function_called", event.data);
          model.save_changes();
        }
      });
    }
    export default { render };
    """  # noqa

    @param.depends("function_called", watch=True)
    def _update_function_called(self):
        print("function_called", self.function_called)
