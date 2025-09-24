from pywinauto.application import Application
import time

def get_pyrometer_temperature(app):
    # app = Application(backend='uia').connect(title = r'BASF TemperaSure 5.7.0.4 Advanced Mode')
    # .start(r"C:\Users\Lab10\Desktop\TemperaSure.exe")
    dlg = app.window(title = r'BASF TemperaSure 5.7.0.4 Advanced Mode')

    # Print all controls recursively
    # dlg.print_control_identifiers()
    # Find the toolbar containing the temperature (ToolBar2)
    toolbar2 = dlg.child_window(control_type="ToolBar", title_re="ToolBar2")

    # Find all Edit controls inside this toolbar
    edit_controls = toolbar2.children(control_type="Edit")

    # If there’s only one Edit control, assume it’s the temperature
    if edit_controls:
        temp_edit = edit_controls[0]
        temperature = temp_edit.get_value()
        # print("Temperature:", temperature)
    else:
        print("No Edit control found in ToolBar2")
    return temperature

def start_pyrometer():

    try:
        app = Application(backend='uia').connect(title = r'BASF TemperaSure 5.7.0.4 Advanced Mode')
        # time.sleep(1)
        app.kill()
        print('killed existing pyrometer window')
    except:
        print('BASF Pyrometer software currently not running')

    print('starting BASF pyrometer software')
    app = Application(backend='uia').start(r"C:\Users\Lab10\Desktop\TemperaSure.exe")
    time.sleep(2)
    main = app.window(title = r'BASF TemperaSure 5.7.0.4 Advanced Mode')
    # dlg = app.window(title="Port setup") 
    # main.print_control_identifiers()
    port = main.child_window(title="Port setup", control_type="Window")
    # dlg = app.window(title="Port setup")  
    # dlg.wait("visible enabled", timeout=2)

    # # Step 2: Click the OK button
    # # Most dialogs have a button literally called "OK"
    # dlg["OK"].click()

    try:
        port.child_window(title="OK", control_type="Button").click()
        print('clicked the ok button')
    except Exception:
        print('was not able to click the ok button')
        pass
    time.sleep(1)
    # Try to click start button
    try:
        main.child_window(title="Start", control_type="Button").click()
        print('clicked the start button')
    except:
        print('was unable to click the start button')

    # Return the pyrometer application
    return app
