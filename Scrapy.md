### Scrapy

#### 框架

![](./picture/spider.png)

1. 爬虫引擎获得初始请求开始爬取。2. 爬虫引擎开始请求调度程序，并准备对下一次请求进行抓取。3. 爬虫调度器返回下一个请求给爬虫引擎。4. 引擎请求发送到下载器，通过下载中间件下载网络数据。5. 一旦下载器完成页面下载，将下载结果返回给爬虫引擎。6. 引擎将下载器的响应通过中间件返回给爬虫进行处理。7. 爬虫处理响应，并通过中间件返回处理后的items，以及新的请求给引擎。 8. 引擎发送处理后的items到项目管道，然后把处理结果返回给调度器，调度器计划处理下一个请求抓取。 9. 重复该过程，直到爬取完所有的`url`请求。

调度器：接收来自引擎的请求并将请求放入队列中，并将事件返回给引擎。

爬虫引擎：负责控制各个组件之间的数据流，当某些操作触发事件后都是通过引擎来处理。

下载器：通过引擎请求下载网络数据并将结果响应给引擎。

Spider：发出请求，并处理引擎返回给它下载器响应数据，以item和规则内的数据请求返回给引擎。

item pipeline：负责处理引擎返回spider解析后的数据，并将数据持久化。

`download middleware`：是引擎和下载器交互组件，以插件的形式存在，可以代替接受请求、处理数据的下载以及将结果响应给引擎。

`spider middleware`：可以代替处理`response`以及返回给引擎items及新的请求集。

当`start_urls`未被指定，会调用`start_requests`，该方法可以用于在爬取数据之前，先进行模拟登陆。

`scrapy.FormRequest`部分参数解析

| 参数          | 解析                                                         |
| ------------- | ------------------------------------------------------------ |
| `fromdata`    | 表单提交的数据                                               |
| `headers`     | 自定义的请求头                                               |
| `meta`        | 向`response`传递数据                                         |
| `errback`     | 错误回调                                                     |
| `callback`    | 回调                                                         |
| `dont_filter` | 如果需要多次提交表单，且`url`一样，必须加此参数。防止被过滤。 |

item pipeline的方法

| 方法           | 作用                       |
| -------------- | -------------------------- |
| `from_crawler` | 可以通过此方法访问settings |
| `open_spider`  | 爬虫启动时调用             |
| `close_spider` | 爬虫关闭时调用             |
| `process_item` | 保存item调用               |

###### Middleware

Middleware
自定义` middleware `需要重写的几个方法

`process_request(request, spider)`
当每个`request`通过下载中间件时，该方法被调用。`process_request()`必须返回其中之一: 返回` None `、返回一个` Response `对象、返回一个` Request `对象或 `raise IgnoreRequest `。如果其返回` None `，`Scrapy`将继续处理该`request`，执行其他的中间件的相应方法，直到合适的下载器处理函数被调用， 该request被执行。如果其返回 Response 对象，`Scrapy`将不会调用 任何 其他的` process_request() `或` process_exception() `方法，或相应地下载函数； 其将返回该 response。 已安装的中间件的` process_response() `方法则会在每个 response 返回时被调用。如果其返回 Request 对象，`Scrapy`则停止调用` process_request`方法并重新调度返回的 request。当新返回的 request 被执行后， 相应地中间件链将会根据下载的 response 被调用。如果其 raise 一个 `IgnoreRequest `异常，则安装的下载中间件的 process_exception() 方法会被调用。如果没有任何一个方法处理该异常， 则 request 的`errback`方法会被调用。如果没有代码处理抛出的异常， 则该异常被忽略且不记录(不同于其他异常那样)。

参数:
request(Request 对象)–处理的 request; spider(Spider 对象)–该 request 对应的 spider

`process_response(request, response, spider)`
process_request() 必须返回以下之一: 返回一个 Response 对象、 返回一个 Request 对象或 raise 一个 `IgnoreRequest `异常。如果其返回一个 Response (可以与传入的response相同，也可以是全新的对象)， 该response会被在链中的其他中间件的 process_response() 方法处理。如果其返回一个 Request 对象，则中间件链停止， 返回的request会被重新调度下载。处理类似于 process_request() 返回request所做的那样。如果其抛出一个` IgnoreRequest `异常，则调用 request 的` errback`。 如果没有代码处理抛出的异常，则该异常被忽略且不记录(不同于其他异常那样)。

参数:
request (Request 对象) – response 所对应的 request; response (Response 对象) – 被处理的 response; spider (Spider 对象) – response 所对应的 spider

`process_exception(request, exception, spider)`
当下载处理器(download handler)或 process_request() (下载中间件)抛出异常时，`Scrapy`调用 process_exception() 。process_exception() 应该返回以下之一: 返回 None 、 一个 Response 对象、或者一个 Request 对象。如果其返回 None ，`Scrapy`将会继续处理该异常，接着调用已安装的其他中间件的 process_exception() 方法，直到所有中间件都被调用完毕，则调用默认的异常处理。如果其返回一个 Response 对象，则已安装的中间件链的 process_response() 方法被调用。`Scrapy`将不会调用任何其他中间件的 process_exception() 方法。如果其返回一个 Request 对象， 则返回的request将会被重新调用下载。这将停止中间件的 process_exception() 方法执行，就如返回一个response的那样。

参数:
request (是 Request 对象) – 产生异常的request; exception (Exception 对象) – 抛出的异常; spider (Spider 对象) – request对应的spider

### CSS Selector

| Selector             | Example            | Description                                                  |
| -------------------- | ------------------ | ------------------------------------------------------------ |
| `.class`             | `.intro`           | selects all elements with `class = "intro"`                  |
| `#id`                | `#firstname`       | selects all elements with `id="firstname"`                   |
| `*`                  | `*`                | selects all elements                                         |
| Element              | `p`                | selects all <p> elements                                     |
| Element, element     | `div, p`           | selects all <div> elements and all < p> elements             |
| Element element      | `div p`            | selects all <p> elements inside <div> elements               |
| Element>element      | `div>p`            | selects all <p> elements where the parent is a <div> element |
| Element+element      | `div+p`            | selects all <p> elements that are placed immediately after <div> elements |
| Element~element      | `p~ul`             | selects every <ul> element that are preceded by a <p> element |
| `[attribute]`        | `[target]`         | selects all elements with a `target` attribute               |
| `[attribute=valuel]` | `[target=_blank]`  | selects all elements with target ="_blank"                   |
| `[attribute~=value]` | `[title~=flower]`  | selects all elements with a title attribute containing the word "flower" |
| `[attribute|=value]` | `[lang|=en]`       | selects all elements with `lang` attribute value starting with "en" |
| `[attribute^=value]` | `a[href^="http"]`  | selects every <a> elements whose `href` attribute value begins with "https" |
| `[attribute$=value]` | `a[href$=".pdf"]`  | selects every <a> elements whose `href` attribute value ends with  ".pdf" |
| `[attribute*=value]` | `a[href*="w3"]`    | selects every <a> elements whose `href` attribute value contains the substring "w3" |
| `:first-of-type`     | `P:first-of-type`  | selects every <p> element that is the first <p> element of its parent |
| `:not(selector)`     | `:not(p)`          | selects every element that is not a <p> element              |
| `:nth-child(n)`      | `P:nth-child(2)`   | selects every <p> element that is the second child of its parent |
| `nth-of-type(n)`     | `P:nth-of-type(2)` | selects every <p> element that is the second <p> element of its parent |

#### BeautifulSoup

对象类型：

1. `tag`：标记，具有属性：`name, attributes`

2. `NavigableString`:`Tag`中的字符串。

3. ` BeautifulSoup`：表示一个文档的全部内容。

4. `comment`：文档的注释部分。

   `find_all(name, attrs, recursive, text, **kwargs)`----`name`:查找所有名字为`name`的标记，可以传入正则表达式，字符串， 列表，`True`和方法--方法接受一个元素参数`Tag`节点，返回`True`表示当前元素匹配并被查找到，反之返回`False`。`kwargs`：如果一个指定名字的参数不是搜索内置的参数名，搜索时会把该参数当作指定`Tag`的属性来搜索。`Text`:搜索文档中的字符串内容。调用`Tag`的`find_all`时，`BeautifulSoup`会检索当前`tag`的所有子孙节点，如果只想搜索`tag`的直接子节点，·可以使用参数`recursive=False`。

   `CSS Selector`:` soup.select`

   