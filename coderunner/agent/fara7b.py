"""The Fara-7B action format (Microsoft's earlier browser agent, on Qwen2.5-VL): its system prompt, its
computer_use tool and its coordinates, copied from github.com/microsoft/fara (src/fara/fara_7b/_prompts.py and
src/fara/qwen_helpers, MIT licence; see THIRD_PARTY_NOTICES.md).

Unlike Fara1.5 (coordinates on a 1000 x 1000 grid, agent/fara.py), Fara-7B states the screen as the screenshot's
size after Qwen2.5-VL's resize (sides rounded to multiples of 28 pixels: 1428 x 896 for a 1440 x 900 page) and
answers in those pixels. Its `type` clicks a point first and may press Enter; `terminate` carries only a status,
so the answer is the reply's text before the call.
"""

from __future__ import annotations

import json
import math

SYSTEM = "You are a helpful assistant."
FACTOR = 28                 # Qwen2.5-VL: 14-pixel patches merged two by two
MIN_PIXELS = 3136
MAX_PIXELS = 12845056
DESCRIPTION = "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is {display_width_px}x{display_height_px}.\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\n* When a separate scrollable container prominently overlays the webpage, if you want to scroll within it, you typically need to mouse_move() over it first and then scroll().\n* If a popup window appears that you want to close, if left_click() on the 'X' or close button doesn't work, try key(keys=['Escape']) to close it.\n* On some search bars, when you type(), you may need to press_enter=False and instead separately call left_click() on the search button to submit the search query. This is especially true of search bars that have auto-suggest popups for e.g. locations\n* For calendar widgets, you usually need to left_click() on arrows to move between months and left_click() on dates to select them; type() is not typically used to input dates there."
PARAMETERS = {'properties': {'action': {'description': 'The action to perform. The available actions are:\n'
                                          '* `key`: Performs key down presses on the arguments passed in '
                                          'order, then performs key releases in reverse order. Includes '
                                          '"Enter", "Alt", "Shift", "Tab", "Control", "Backspace", "Delete", '
                                          '"Escape", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", '
                                          '"PageDown", "PageUp", "Shift", etc.\n'
                                          '* `type`: Type a string of text on the keyboard.\n'
                                          '* `mouse_move`: Move the cursor to a specified (x, y) pixel '
                                          'coordinate on the screen.\n'
                                          '* `left_click`: Click the left mouse button.\n'
                                          '* `scroll`: Performs a scroll of the mouse scroll wheel.\n'
                                          '* `visit_url`: Visit a specified URL.\n'
                                          '* `web_search`: Perform a web search with a specified query.\n'
                                          '* `history_back`: Go back to the previous page in the browser '
                                          'history.\n'
                                          '* `pause_and_memorize_fact`: Pause and memorize a fact for future '
                                          'reference.\n'
                                          '* `wait`: Wait specified seconds for the change to happen.\n'
                                          '* `terminate`: Terminate the current task and report its '
                                          'completion status.',
                           'enum': ['key',
                                    'type',
                                    'mouse_move',
                                    'left_click',
                                    'scroll',
                                    'visit_url',
                                    'web_search',
                                    'history_back',
                                    'pause_and_memorize_fact',
                                    'wait',
                                    'terminate'],
                           'type': 'string'},
                'keys': {'description': 'Required only by `action=key`.', 'type': 'array'},
                'text': {'description': 'Required only by `action=type`.', 'type': 'string'},
                'press_enter': {'description': 'Whether to press the Enter key after typing. Required only '
                                               'by `action=type`.',
                                'type': 'boolean'},
                'delete_existing_text': {'description': 'Whether to delete existing text before typing. '
                                                        'Required only by `action=type`.',
                                         'type': 'boolean'},
                'coordinate': {'description': '(x, y): The x (pixels from the left edge) and y (pixels from '
                                              'the top edge) coordinates to move the mouse to. Required only '
                                              'by `action=left_click`, `action=mouse_move`, and '
                                              '`action=type`.',
                               'type': 'array'},
                'pixels': {'description': 'The amount of scrolling to perform. Positive values scroll up, '
                                          'negative values scroll down. Required only by `action=scroll`.',
                           'type': 'number'},
                'url': {'description': 'The URL to visit. Required only by `action=visit_url`.',
                        'type': 'string'},
                'query': {'description': 'The query to search for. Required only by `action=web_search`.',
                          'type': 'string'},
                'fact': {'description': 'The fact to remember for the future. Required only by '
                                        '`action=pause_and_memorize_fact`.',
                         'type': 'string'},
                'time': {'description': 'The seconds to wait. Required only by `action=wait`.',
                         'type': 'number'},
                'status': {'description': 'The status of the task. Required only by `action=terminate`.',
                           'type': 'string',
                           'enum': ['success', 'failure']}},
 'required': ['action'],
 'type': 'object'}
TEMPLATE = "You are a web automation agent that performs actions on websites to fulfill user requests by calling various tools.\n* You should stop execution at Critical Points. A Critical Point would be encountered in tasks like 'Checkout', 'Book', 'Purchase', 'Call', 'Email', 'Order', etc where a binding transaction/agreement would require the user's permission/personal or sensitive information (name, email, credit card, address, payment information, resume, etc) in order to complete a transaction (purchase, reservation, sign-up etc), or to communicate in a way that a human would be expected to do (call, email, apply to a job, etc).\n* Solve the task as far as you can up until a Critical Point:\n    - For example, if the task is to \"call a restaurant to make a reservation\", you should not actually make the call but should navigate to the restaurant's page and find the phone number.\n    - Similarly, if the task is to \"order new size 12 running shoes\" you should not actually place the order but should instead search for the right shoes that meet the criteria and add them to the cart.\n    - Some tasks, like answering questions, may not encounter a Critical Point at all.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tool_descs}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>"


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """Qwen2.5-VL's resize: sides rounded to multiples of FACTOR, area kept within MIN_PIXELS and MAX_PIXELS."""
    h, w = max(FACTOR, round(height / FACTOR) * FACTOR), max(FACTOR, round(width / FACTOR) * FACTOR)
    if h * w > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h, w = math.floor(height / beta / FACTOR) * FACTOR, math.floor(width / beta / FACTOR) * FACTOR
    elif h * w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h, w = math.ceil(height * beta / FACTOR) * FACTOR, math.ceil(width * beta / FACTOR) * FACTOR
    return h, w


def system_prompt(width: float, height: float) -> str:
    """The system message for a page of width x height CSS pixels (its screenshot is taken at that size)."""
    resized_h, resized_w = smart_resize(int(height), int(width))
    function = {"name": "computer_use",
                "description": DESCRIPTION.format(display_width_px=resized_w, display_height_px=resized_h),
                "parameters": PARAMETERS}
    tool = json.dumps({"type": "function", "function": function}, ensure_ascii=False)
    return SYSTEM + "\n\n" + TEMPLATE.format(tool_descs=tool)


def to_pixels(coordinate, width: float, height: float) -> tuple[float, float] | None:
    """A point in the resized screenshot's pixels as CSS pixels of the page's viewport."""
    if not (isinstance(coordinate, (list, tuple)) and len(coordinate) == 2) or not width or not height:
        return None
    try:
        x, y = float(coordinate[0]), float(coordinate[1])
    except (TypeError, ValueError):
        return None
    resized_h, resized_w = smart_resize(int(height), int(width))
    return min(max(x, 0), resized_w) * width / resized_w, min(max(y, 0), resized_h) * height / resized_h
