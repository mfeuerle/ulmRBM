{{ fullname | escape | underline}}

{% set m = ['__matmul__', '__rmatmul__', 
'__mul__', '__rmul__', 
'__add__', '__radd__', 
'__sub__', '__rsub__', 
'__neg__', 
'__call__', 
'__truediv__', '__floordiv__',
'__mod__', '__rmod__',
'__pow__',
'__getitem__', '__setitem__', '__delitem__',
'__iter__',
'__len__'] %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   
   {% for item in all_attributes %}
         {%- if not item.startswith('_') %}
   .. autoattribute:: {{ item }}
         {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

  {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: {{ objname }}
   {% for item in all_methods %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}


   {% set ns = namespace(overload_methods=[]) %}
   {% for item in all_methods %}
      {% if item in m %}
         {% set ns.overload_methods = ns.overload_methods + [item] %}
      {% endif %}
   {% endfor %}
   
   {% if ns.overload_methods %}
   .. rubric:: Operator Overloads

   .. autosummary::
      :toctree: {{ objname }}
   {% for item in ns.overload_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}

   {% endif %}
  {% endblock %}

.. 
   .. collapse:: More Dunder Methods

      .. autosummary::
         :toctree: {{ objname }}
      {% for item in all_methods %}
         {%- if item.startswith('__') and item.endswith('__') and not item in m %}
         ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}