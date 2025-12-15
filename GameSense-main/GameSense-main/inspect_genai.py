import importlib, sys
try:
    m = importlib.import_module('google.generativeai')
except Exception as e:
    print('IMPORT_ERROR', e)
    sys.exit(0)
print('MODULE_FILE', getattr(m, '__file__', None))
attrs = sorted(dir(m))
print('HAS_GenerativeModel', 'GenerativeModel' in attrs)
print('HAS_generate_text', 'generate_text' in attrs)
print('HAS_generate', 'generate' in attrs)
print('HAS_configure', 'configure' in attrs)
print('SAMPLE_GENERATE_ATTRS', [a for a in attrs if a.lower().startswith('generat')][:50])
print('VERSION', getattr(m, '__version__', None))
