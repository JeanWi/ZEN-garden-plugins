.. _implementation.implementing_plugins:

####################################
Implementing Your Own Plugin
####################################

This guide walks you through creating a plugin for ZEN-garden from scratch.
Don't worry if you're new to Python—we'll keep it simple!

**Two ways to develop a plugin:**

1. **With this repository (recommended for beginners):** Fork this repository and develop your plugin here
2. **In a separate repository:** Create your own plugin repository and install it as a Python package


Getting started
---------------

**Fork the template**

Start by using the ``plugin_template/`` directory:

1. Fork this repository on GitHub
2. Copy ``plugin_template/`` and rename it (e.g., ``my_awesome_plugin/``)
3. Edit the files inside

**Key files in your plugin:**

- ``plugin.py`` — Your plugin's main code (required!)
- ``pyproject.toml`` — Package information
- ``tests/`` — Tests for your plugin
- ``docs/`` — Documentation


Step 1: Set up ``plugin.py``
-----------------------------

Every ``plugin.py`` **must** have:

1. **A `Config` class** with default settings:

   .. code-block:: python

       from zen_garden import ConfigBase

       class Config(ConfigBase):
           """Configuration for my plugin."""
           my_setting: str = "default_value"

2. **One or more functions** decorated with ``@EventPublisher.register``:

   .. code-block:: python

       from zen_garden import Event, EventPublisher

       @EventPublisher.register(Event.after_model_schema_creation)
       def my_plugin_logic(model_schema):
           """This runs at a specific point in ZEN-garden's workflow."""
           # Your code goes here!
           pass

**How it works:**

- ZEN-garden has a workflow with specific points where plugins can "hook in"
- Each hook is an "event" (e.g., ``after_model_schema_creation``)
- Your function is called automatically when ZEN-garden reaches that event
- ZEN-garden passes relevant objects to your function (e.g., the model schema)

**Available events:**

See ``zen_garden.plugin_system.events.Event`` for all available events. Common ones:

- ``after_model_schema_creation`` — After the model structure is built
- ``before_optimization`` — Just before optimization starts
- ``after_optimization`` — After optimization completes

Each event passes specific keyword arguments. Check the ZEN-garden source or docstrings
to see what each event provides.


Step 2: Access your configuration
----------------------------------

Inside your plugin function, access the settings passed from ZEN-garden:

.. code-block:: python

    @EventPublisher.register(Event.after_model_schema_creation)
    def my_plugin_logic(model_schema):
        # Get your plugin's config from ZEN-garden
        config = model_schema.config.plugins["my_plugin"]
        my_value = config["my_setting"]
        print(f"Using setting: {my_value}")


Step 3: Update ``pyproject.toml``
----------------------------------

Tell Python that your plugin is a ZEN-garden plugin:

.. code-block:: toml

    [project]
    name = "zen_garden_my_plugin"
    version = "0.1.0"
    description = "My awesome ZEN-garden plugin"

    [project.entry-points."zen_garden.plugins"]
    my_plugin = "zen_garden_plugins.my_plugin"

The entry point name (``my_plugin``) is what users write in their ``config.yaml``.


Step 4: Test your plugin
------------------------

Create a ``tests/`` directory with tests:

.. code-block:: python

    # tests/test_my_plugin.py
    import pytest
    from zen_garden_plugins.my_plugin.plugin import Config

    def test_config_has_defaults():
        config = Config()
        assert config.my_setting == "default_value"


Step 5: Document your plugin
-----------------------------

Create ``docs/files/available_plugins/my_plugin/`` with:

- ``my_plugin.rst`` — Overview and usage guide
- Any other documentation users need

See ``template_plugin.rst`` for an example.


Step 6: Install and test
------------------------

Test locally by installing in editable mode:

.. code-block:: shell

    pip install -e .

This lets you test changes immediately without reinstalling.


Step 7: Configure in ZEN-garden
-------------------------------

Users activate your plugin in their ``config.yaml``:

.. code-block:: yaml

    plugins:
      my_plugin:
        my_setting: "custom_value"

That's it! ZEN-garden will:

1. Find your installed plugin
2. Load your ``Config`` with the custom value
3. Call your decorated functions at the right times


See the template plugin
-----------------------

The template plugin in this repository is a working example. Look at:

- ``zen_garden_plugins/plugin_template/plugin.py`` — Simple example code
- ``docs/files/available_plugins/template_plugin/`` — Example documentation

Use it as a reference when building your own plugin!
