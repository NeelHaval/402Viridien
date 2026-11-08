import AppKit
import WebKit

let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
class Runner: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
    var web: WKWebView!
    var window: NSWindow!
    func start() {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        config.userContentController.add(self, name: "report")
        web = WKWebView(frame: NSRect(x:0,y:0,width:1500,height:1000), configuration:config)
        web.navigationDelegate = self
        window = NSWindow(contentRect: web.frame, styleMask: [.borderless], backing:.buffered, defer:false)
        window.contentView = web
        window.orderFrontRegardless()
        web.load(URLRequest(url:URL(string:CommandLine.arguments[1])!))
        DispatchQueue.main.asyncAfter(deadline:.now()+50) { print("TIMEOUT"); exit(2) }
    }
    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        do {
            let code = try String(contentsOfFile:CommandLine.arguments[2],encoding:.utf8)
            web.evaluateJavaScript(code + ";void 0;") { _,err in if let err=err {print("JS ERROR \(err)");exit(3)} }
        } catch { print(error);exit(4) }
    }
    func userContentController(_ controller: WKUserContentController,didReceive message: WKScriptMessage) {
        if let body = message.body as? [String:Any], body["action"] as? String == "drag",
           let x1 = body["x1"] as? Double, let y1 = body["y1"] as? Double,
           let x2 = body["x2"] as? Double, let y2 = body["y2"] as? Double {
            let hold = (body["holdMs"] as? Double ?? 0) / 1000
            for i in 0...12 {
                DispatchQueue.main.asyncAfter(deadline:.now()+Double(i)*0.04+(i == 12 ? hold : 0)) {
                    let t = max(0,min(1,Double(i-1)/10))
                    let type: NSEvent.EventType = i == 0 ? .leftMouseDown : (i == 12 ? .leftMouseUp : .leftMouseDragged)
                    let point = NSPoint(x:x1+(x2-x1)*t,y:self.web.bounds.height-(y1+(y2-y1)*t))
                    if let event = NSEvent.mouseEvent(with:type,location:point,modifierFlags:[],timestamp:ProcessInfo.processInfo.systemUptime,windowNumber:self.window.windowNumber,context:nil,eventNumber:i,clickCount:1,pressure:1) {
                        self.window.sendEvent(event)
                    }
                }
            }
            return
        }
        print(message.body)
        let config=WKSnapshotConfiguration()
        web.takeSnapshot(with:config) { image,error in
            if let image=image,let tiff=image.tiffRepresentation,let rep=NSBitmapImageRep(data:tiff),let png=rep.representation(using:.png,properties:[:]) {
                try? png.write(to:URL(fileURLWithPath:CommandLine.arguments[3]))
            }
            exit(error == nil ? 0:5)
        }
    }
}
let runner=Runner()
runner.start()
app.run()
